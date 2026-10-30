"""
basic_transformer.py
Implementation of the Vanilla Sequence-to-Sequence Transformer
Encoder-Decoder Architecture (Vaswani et al., 2017)
"""

import math
import time
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .dataset_loader import Vocabulary, SPECIAL_TOKENS


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # 1) Linear projections and split heads: [B, H, L, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Scaled dot-product: [B, H, L_q, L_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Broadcast mask over head dimension if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 3) Attention-weighted sum: [B, H, L_q, d_k] -> [B, L_q, d_model]
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Sub-layer 1: Self-attention
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Sub-layer 2: Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer with Masked Self-Attention and Cross-Attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Sub-layer 1: Masked Self-attention
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Sub-layer 2: Cross-attention (Query from decoder, Key/Value from encoder memory)
        cross_attn_out = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Sub-layer 3: Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class BasicTransformerSeq2Seq(nn.Module):
    """
    Complete Vanilla Encoder-Decoder Transformer Architecture.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = SPECIAL_TOKENS["<pad>"],
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)]
        )

        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # src: [batch_size, src_len] -> [batch_size, 1, 1, src_len]
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # tgt: [batch_size, tgt_len]
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_t]
        tgt_len = tgt.size(1)
        # Causal triangular subsequent mask: [1, 1, L_t, L_t]
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return tgt_pad_mask & causal_mask

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def translate(
        self,
        src_text: str,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = 50,
        temperature: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> str:
        self.eval()
        src_tokens = src_vocab.encode(src_text, add_special_tokens=True)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
        src_mask = self.make_src_mask(src_tensor)

        memory = self.encode(src_tensor, src_mask)

        ys = torch.tensor([[SPECIAL_TOKENS["<sos>"]]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(ys)
            out = self.decode(ys, memory, tgt_mask, src_mask)
            logits = self.generator(out[:, -1, :])  # [1, vocab_size]

            if temperature <= 0.001:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == SPECIAL_TOKENS["<eos>"]:
                break

            ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

        pred_ids = ys.squeeze(0).tolist()[1:]  # skip <sos>
        return tgt_vocab.decode(pred_ids, remove_special_tokens=True)


class ParallelTextDataset(Dataset):
    """PyTorch Dataset for Parallel Text."""

    def __init__(self, data: List[dict], src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        item = self.data[idx]
        src_ids = self.src_vocab.encode(item["en"], add_special_tokens=True)
        tgt_ids = self.tgt_vocab.encode(item["fr"], add_special_tokens=True)
        return src_ids, tgt_ids


def collate_fn(batch):
    """Pads batches dynamically to the maximum sequence length in the batch."""
    src_list, tgt_list = zip(*batch)
    pad_idx = SPECIAL_TOKENS["<pad>"]

    src_lens = [len(s) for s in src_list]
    tgt_lens = [len(t) for t in tgt_list]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    padded_src = [s + [pad_idx] * (max_src - len(s)) for s in src_list]
    padded_tgt = [t + [pad_idx] * (max_tgt - len(t)) for t in tgt_list]

    return (
        torch.tensor(padded_src, dtype=torch.long),
        torch.tensor(padded_tgt, dtype=torch.long),
    )


def train_basic_transformer(
    model: BasicTransformerSeq2Seq,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 5e-4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[float], List[float], float]:
    """Trains the Basic Transformer and records training time and loss curves."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<pad>"])

    train_losses, val_losses = [], []
    start_time = time.time()

    print(f"\n[Basic Transformer] Starting training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            # Teacher forcing: input is tgt[:, :-1], label is tgt[:, 1:]
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_in)  # [B, L_t-1, V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        if epoch % 2 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"[Basic Transformer] Training finished in {total_time:.2f} seconds.")
    return train_losses, val_losses, total_time

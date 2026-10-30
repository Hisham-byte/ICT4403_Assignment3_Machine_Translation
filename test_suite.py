"""
test_suite.py
Automated End-to-End Verification and Unit Test Suite
Validates all components of ICT4403 Assignment 3:
1. Dataset Loader & Preprocessing
2. Basic Transformer Architecture & Tensor Shapes
3. Forward Pass, Loss, Backward Step & Translation Decoding
4. mT5 Translator & Sequence-to-Sequence Formatting
5. LLaMA & Gemma Decoder-Only In-Context Prompting
6. BLEU and chrF Metric Computations
7. Ethical AI Evaluation Probes
8. End-to-End Pipeline Output Files (CSVs and PNG Plots)
9. Word Document (.docx) and PDF Generation
"""

import os
import sys
import time
import math
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tabulate import tabulate

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.dataset_loader import (
    load_opus_books_dataset,
    build_vocabularies,
    normalize_text,
    basic_tokenize,
    Vocabulary,
    SPECIAL_TOKENS
)
from src.basic_transformer import (
    BasicTransformerSeq2Seq,
    PositionalEncoding,
    MultiHeadAttention,
    ParallelTextDataset,
    collate_fn,
    train_basic_transformer
)
from src.mt5_translator import MT5Translator
from src.decoder_models import LLaMATranslator, GemmaTranslator
from src.metrics import (
    calculate_sentence_bleu,
    calculate_corpus_bleu,
    calculate_corpus_chrf,
    evaluate_model_performance
)
from src.ethical_evaluator import evaluate_ethical_probes_qualitatively


def test_section(title: str):
    print("\n" + "=" * 70)
    print(f" [TEST] {title}")
    print("=" * 70)


def run_all_tests():
    total_passed = 0
    total_tests = 0

    # -------------------------------------------------------------
    # Test 1: Preprocessing and Tokenization
    # -------------------------------------------------------------
    test_section("1. Text Preprocessing & Tokenization")
    total_tests += 1
    raw_str = "  L'intelligence  artificielle transforme “la société” moderne!  "
    norm_str = normalize_text(raw_str)
    tokens = basic_tokenize(norm_str)
    assert "l'intelligence" in tokens or "intelligence" in tokens, "Tokenization contraction check failed"
    assert norm_str == 'L\'intelligence artificielle transforme "la société" moderne!', f"Unexpected norm: {norm_str}"
    print(f"  [PASS] normalize_text: '{raw_str.strip()}' -> '{norm_str}'")
    print(f"  [PASS] basic_tokenize tokens: {tokens[:5]}... (Total: {len(tokens)})")
    total_passed += 1

    # -------------------------------------------------------------
    # Test 2: Dataset Loading & Split Verification
    # -------------------------------------------------------------
    test_section("2. Dataset Loading & Partitioning")
    total_tests += 1
    train_data, val_data, test_data = load_opus_books_dataset(max_samples=2500, data_dir=os.path.join(PROJECT_DIR, "data"))
    assert len(train_data) > 0, "Train dataset empty"
    assert len(val_data) > 0, "Val dataset empty"
    assert len(test_data) > 0, "Test dataset empty"
    print(f"  [PASS] Splits loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    src_vocab, tgt_vocab = build_vocabularies(train_data, min_freq=2)
    assert len(src_vocab) > 50, "Source vocab too small"
    assert len(tgt_vocab) > 50, "Target vocab too small"
    print(f"  [PASS] Vocabularies built: Source (EN)={len(src_vocab)}, Target (FR)={len(tgt_vocab)}")
    total_passed += 1

    # -------------------------------------------------------------
    # Test 3: Transformer Architectural Components & Tensor Shapes
    # -------------------------------------------------------------
    test_section("3. Basic Transformer Layers & Attention Shapes")
    total_tests += 1
    d_model = 64
    n_heads = 4
    batch_size = 2
    seq_len = 10

    # Test Positional Encoding
    pos_enc = PositionalEncoding(d_model=d_model, max_len=100)
    dummy_x = torch.zeros(batch_size, seq_len, d_model)
    pos_out = pos_enc(dummy_x)
    assert pos_out.shape == (batch_size, seq_len, d_model), f"Shape mismatch: {pos_out.shape}"
    print(f"  [PASS] PositionalEncoding output shape: {pos_out.shape}")

    # Test Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    attn_out = mha(dummy_x, dummy_x, dummy_x)
    assert attn_out.shape == (batch_size, seq_len, d_model), f"Shape mismatch: {attn_out.shape}"
    print(f"  [PASS] MultiHeadAttention output shape: {attn_out.shape}")
    total_passed += 1

    # -------------------------------------------------------------
    # Test 4: End-to-End Seq2Seq Transformer Forward Pass & Step
    # -------------------------------------------------------------
    test_section("4. Full Seq2Seq Transformer Model & Forward Pass")
    total_tests += 1
    model = BasicTransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        dropout=0.1
    )
    param_count = model.count_parameters()
    assert param_count > 100_000, f"Unexpectedly small parameter count: {param_count}"
    print(f"  [PASS] Model initialized with {param_count:,} parameters.")

    # Collate batch and forward
    dataset = ParallelTextDataset(train_data[:4], src_vocab, tgt_vocab)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for src, tgt in loader:
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in)
        assert logits.shape == (src.size(0), tgt_in.size(1), len(tgt_vocab)), f"Logits shape mismatch: {logits.shape}"
        
        # Loss computation
        criterion = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<pad>"])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        assert not torch.isnan(loss), "Loss computed is NaN!"
        loss.backward()
        print(f"  [PASS] Forward & Backward pass succeeded. Batch Loss: {loss.item():.4f}")
        break

    # Test Translation Decoding (Greedy & Temperature)
    sample_en = "The morning sun lit up the ancient stone castle."
    pred_greedy = model.translate(sample_en, src_vocab, tgt_vocab, temperature=0.0)
    pred_sampled = model.translate(sample_en, src_vocab, tgt_vocab, temperature=0.7)
    assert isinstance(pred_greedy, str) and len(pred_greedy) > 0, "Greedy translation empty"
    assert isinstance(pred_sampled, str) and len(pred_sampled) > 0, "Sampled translation empty"
    print(f"  [PASS] Greedy Translation: '{pred_greedy}'")
    print(f"  [PASS] Sampled Translation (T=0.7): '{pred_sampled}'")
    total_passed += 1

    # -------------------------------------------------------------
    # Test 5: mT5 Sequence-to-Sequence Wrapper
    # -------------------------------------------------------------
    test_section("5. mT5 Translator Wrapper")
    total_tests += 1
    mt5 = MT5Translator(model_name="google/mt5-small", device=torch.device("cpu"))
    mt5.load_model()
    assert mt5.param_count >= 300_000_000, "mT5 param count incorrect"
    ft_time = mt5.fine_tune(train_data[:10], val_data[:5], epochs=1)
    trans = mt5.translate("The morning sun lit up the castle.")
    assert isinstance(trans, str) and len(trans) > 0, "mT5 translation failed"
    print(f"  [PASS] mT5 Model Verified. Parameters: {mt5.param_count:,} | Fine-tune time: {ft_time:.1f}s")
    total_passed += 1

    # -------------------------------------------------------------
    # Test 6: LLaMA & Gemma Decoder-Only Prompting
    # -------------------------------------------------------------
    test_section("6. LLaMA & Gemma Decoder-Only Translation Prompting")
    total_tests += 1
    llama = LLaMATranslator(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=torch.device("cpu"))
    llama_prompt = llama.build_prompt("Knowledge is power.")
    assert "<|system|>" in llama_prompt and "<|user|>" in llama_prompt, "LLaMA prompt formatting error"
    print(f"  [PASS] LLaMA Prompt Template Validated (Length: {len(llama_prompt)} chars)")

    gemma = GemmaTranslator(model_name="google/gemma-2b-it", device=torch.device("cpu"))
    gemma_prompt = gemma.build_prompt("Knowledge is power.")
    assert "<start_of_turn>user" in gemma_prompt and "<start_of_turn>model" in gemma_prompt, "Gemma prompt formatting error"
    print(f"  [PASS] Gemma Prompt Template Validated (Length: {len(gemma_prompt)} chars)")
    total_passed += 1

    # -------------------------------------------------------------
    # Test 7: Evaluation Metrics (BLEU & chrF)
    # -------------------------------------------------------------
    test_section("7. Evaluation Metrics (BLEU & chrF)")
    total_tests += 1
    hypotheses = ["le soleil du matin illuminait l'ancien château de pierre."]
    references = ["le soleil du matin illuminait l'ancien château de pierre."]
    bleu_perfect = calculate_corpus_bleu(hypotheses, references)
    chrf_perfect = calculate_corpus_chrf(hypotheses, references)
    assert bleu_perfect >= 99.0, f"Expected 100 for identical strings, got {bleu_perfect}"
    assert chrf_perfect >= 99.0, f"Expected 100 for identical strings, got {chrf_perfect}"

    hyp_diff = ["la lumière matinale éclairait la bâtisse ancienne."]
    bleu_diff = calculate_corpus_bleu(hyp_diff, references)
    chrf_diff = calculate_corpus_chrf(hyp_diff, references)
    assert 0 <= bleu_diff <= 100.0, "BLEU range invalid"
    assert 0 <= chrf_diff <= 100.0, "chrF range invalid"
    print(f"  [PASS] Perfect Match -> BLEU: {bleu_perfect:.2f} | chrF: {chrf_perfect:.2f}")
    print(f"  [PASS] Paraphrase Match -> BLEU: {bleu_diff:.2f} | chrF: {chrf_diff:.2f}")
    total_passed += 1

    # -------------------------------------------------------------
    # Test 8: Ethical AI Probes
    # -------------------------------------------------------------
    test_section("8. Responsible & Ethical AI Probes")
    total_tests += 1
    ethical_probes = evaluate_ethical_probes_qualitatively()
    assert len(ethical_probes) >= 4, f"Expected at least 4 probes, found {len(ethical_probes)}"
    for p in ethical_probes:
        assert "Probe" in p and "Ethical Vulnerability Observed" in p and "Mitigation Strategy" in p
    print(f"  [PASS] Evaluated {len(ethical_probes)} ethical failure-mode probes (bias, hallucination, cultural idioms).")
    total_passed += 1

    # -------------------------------------------------------------
    # Test 9: Output Files & Figures Verification
    # -------------------------------------------------------------
    test_section("9. Generated Artifacts & Figures Verification")
    total_tests += 1
    required_files = [
        os.path.join(PROJECT_DIR, "results", "model_comparison_metrics.csv"),
        os.path.join(PROJECT_DIR, "results", "temperature_analysis.csv"),
        os.path.join(PROJECT_DIR, "results", "sample_translations.csv"),
        os.path.join(PROJECT_DIR, "results", "ethical_analysis_samples.csv"),
        os.path.join(PROJECT_DIR, "plots", "bleu_chrf_comparison.png"),
        os.path.join(PROJECT_DIR, "plots", "inference_latency_comparison.png"),
        os.path.join(PROJECT_DIR, "plots", "temperature_vs_bleu_diversity.png"),
        os.path.join(PROJECT_DIR, "HishamAdil_230953434_FGAI_Assignment3.docx"),
        os.path.join(PROJECT_DIR, "HishamAdil_230953434_FGAI_Assignment3.pdf"),
        os.path.join(PROJECT_DIR, "assignment3_machine_translation.ipynb"),
        os.path.join(PROJECT_DIR, "README.md"),
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required deliverable: {rf}"
        sz = os.path.getsize(rf)
        assert sz > 0, f"Deliverable is empty: {rf}"
        print(f"  [PASS] Found: {os.path.basename(rf)} ({sz:,} bytes)")
    total_passed += 1

    # Final summary
    print("\n" + "=" * 70)
    print(f" ALL TESTS PASSED: {total_passed}/{total_tests} (100% Success Rate)")
    print("=" * 70)
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)

"""
metrics.py
Evaluation Metrics for Machine Translation
1. BLEU (Bilingual Evaluation Understudy)
2. chrF (Character n-gram F-score)
3. Inference Latency & Throughput
4. Parameter Counts & Architecture Specifications
"""

import math
from collections import Counter
from typing import List, Dict, Any


def compute_ngram_counts(tokens: List[str], n: int) -> Counter:
    """Extracts n-gram counts from token list."""
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def calculate_sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """
    Standard Sentence BLEU score with smoothing and Brevity Penalty (Papineni et al., 2002).
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    c = len(hyp_tokens)
    r = len(ref_tokens)

    if c == 0:
        return 0.0

    # Brevity penalty
    bp = 1.0 if c > r else math.exp(1.0 - float(r) / float(c))

    log_prec_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = compute_ngram_counts(hyp_tokens, n)
        ref_ngrams = compute_ngram_counts(ref_tokens, n)

        clipped_matches = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        total_ngrams = max(1, c - n + 1)
        # Smoothing (add 0.1 to avoid zero log)
        prec = (clipped_matches + 0.1) / (total_ngrams + 0.1)
        log_prec_sum += (1.0 / max_n) * math.log(prec)

    return bp * math.exp(log_prec_sum) * 100.0


def calculate_corpus_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Calculates Corpus BLEU score via SacreBLEU or mathematical fallback."""
    try:
        import sacrebleu
        # sacrebleu expects references as a list of reference streams: [[ref1, ref2, ...]]
        res = sacrebleu.corpus_bleu(hypotheses, [references])
        return round(res.score, 2)
    except Exception:
        # Robust sentence-averaged fallback
        scores = [calculate_sentence_bleu(h, r) for h, r in zip(hypotheses, references)]
        return round(sum(scores) / max(1, len(scores)), 2)


def calculate_corpus_chrf(hypotheses: List[str], references: List[str], beta: int = 2) -> float:
    """Calculates chrF score via SacreBLEU or character n-gram F-score fallback."""
    try:
        import sacrebleu
        res = sacrebleu.corpus_chrf(hypotheses, [references], beta=beta)
        return round(res.score, 2)
    except Exception:
        # Popovic (2015) character n-gram F-score implementation (n=1..6)
        total_f_scores = []
        for hyp, ref in zip(hypotheses, references):
            h_clean = "".join(hyp.lower().split())
            r_clean = "".join(ref.lower().split())

            if not h_clean or not r_clean:
                total_f_scores.append(0.0)
                continue

            precisions, recalls = [], []
            for n in range(1, 7):
                h_ngrams = Counter([h_clean[i : i + n] for i in range(len(h_clean) - n + 1)])
                r_ngrams = Counter([r_clean[i : i + n] for i in range(len(r_clean) - n + 1)])

                matches = sum(min(cnt, r_ngrams[ng]) for ng, cnt in h_ngrams.items())
                p = matches / max(1, len(h_clean) - n + 1)
                r = matches / max(1, len(r_clean) - n + 1)
                precisions.append(p)
                recalls.append(r)

            avg_p = sum(precisions) / len(precisions)
            avg_r = sum(recalls) / len(recalls)

            if avg_p + avg_r == 0:
                f_score = 0.0
            else:
                beta_sq = beta ** 2
                f_score = (1 + beta_sq) * (avg_p * avg_r) / (beta_sq * avg_p + avg_r)
            total_f_scores.append(f_score * 100.0)

        return round(sum(total_f_scores) / max(1, len(total_f_scores)), 2)


def evaluate_model_performance(
    model_name: str,
    architecture_type: str,
    param_count: int,
    training_time_sec: float,
    inference_latency_ms: float,
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, Any]:
    """Compiles complete quantitative metrics for a single model."""
    bleu = calculate_corpus_bleu(hypotheses, references)
    chrf = calculate_corpus_chrf(hypotheses, references)

    return {
        "Model": model_name,
        "Architecture": architecture_type,
        "Parameters (M)": round(param_count / 1e6, 1),
        "Training / FT Time (s)": round(training_time_sec, 1) if training_time_sec > 0 else "N/A (Prompting)",
        "Inference Latency (ms/sent)": round(inference_latency_ms, 2),
        "BLEU Score": bleu,
        "chrF Score": chrf,
    }

"""
Project 56: Optical Character Recognition with Transformers (TrOCR)
===================================================================
This project implements end-to-end Optical Character Recognition (OCR) using
Microsoft's TrOCR (Transformer OCR) architecture.

TrOCR replaces traditional multi-stage pipelines (CNN detection + RNN + CTC) with
a pure Vision-Encoder-Decoder Transformer:
- Encoder: Pre-trained Vision Transformer (DeiT)
- Decoder: Pre-trained Autoregressive Language Model (MiniLM / RoBERTa)
- Pretrained weights: microsoft/trocr-small-printed (61.6M parameters)
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Tuple

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Deterministic seeding
torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """Computes the Levenshtein edit distance between two sequences (chars or words)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]


def compute_cer(ref: str, hyp: str, case_sensitive: bool = True) -> float:
    """Character Error Rate (CER)."""
    if not case_sensitive:
        ref = ref.lower()
        hyp = hyp.lower()
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(dist / max(len(ref_chars), 1))


def compute_wer(ref: str, hyp: str, case_sensitive: bool = True) -> float:
    """Word Error Rate (WER)."""
    if not case_sensitive:
        ref = ref.lower()
        hyp = hyp.lower()
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist / max(len(ref_words), 1))


class TrOCREngine:
    """Encapsulates the TrOCR model, image preprocessor, and text decoder."""

    def __init__(self, model_id: str = "microsoft/trocr-small-printed", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading TrOCR processor & VisionEncoderDecoderModel ({model_id}) on {self.device.upper()}...")
        t0 = time.time()
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        t_load = time.time() - t0

        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model successfully initialized in {t_load:.2f}s! Total parameters: {self.num_params:,} ({self.num_params/1e6:.1f}M)")

    def recognize_text(self, image: Image.Image, max_new_tokens: int = 64) -> Dict[str, Any]:
        """Runs Transformer OCR on an input image."""
        t0 = time.time()
        pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens)

        latency_ms = (time.time() - t0) * 1000.0
        predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        chars_per_sec = (len(predicted_text) / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0
        return {
            "text": predicted_text,
            "latency_ms": latency_ms,
            "chars_per_sec": chars_per_sec,
            "num_tokens": generated_ids.shape[1]
        }


def evaluate_benchmark(engine: TrOCREngine, dataset_path: str) -> Dict[str, Any]:
    """Runs OCR benchmark across curated synthetic test samples."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"\nEvaluating TrOCR on {len(samples)} benchmark document samples...")
    print("=" * 95)
    print(f"{'ID':<6} | {'Category':<10} | {'Raw CER':<8} | {'Norm CER':<8} | {'Latency':<8} | {'Prediction'}")
    print("-" * 95)

    results = []
    category_metrics = {}
    raw_cers, norm_cers = [], []
    raw_wers, norm_wers = [], []
    latencies = []
    exact_matches = 0
    norm_exact_matches = 0

    for s in samples:
        img_path = os.path.join(IMAGES_DIR, s["image"])
        img = Image.open(img_path).convert("RGB")
        ref_text = s["ground_truth"]
        cat = s.get("category", "general")

        pred_info = engine.recognize_text(img)
        hyp_text = pred_info["text"]
        
        raw_cer = compute_cer(ref_text, hyp_text, case_sensitive=True)
        raw_wer = compute_wer(ref_text, hyp_text, case_sensitive=True)
        norm_cer = compute_cer(ref_text, hyp_text, case_sensitive=False)
        norm_wer = compute_wer(ref_text, hyp_text, case_sensitive=False)
        
        is_exact = (ref_text.strip() == hyp_text.strip())
        is_norm_exact = (ref_text.strip().lower() == hyp_text.strip().lower())

        if is_exact:
            exact_matches += 1
        if is_norm_exact:
            norm_exact_matches += 1

        raw_cers.append(raw_cer)
        norm_cers.append(norm_cer)
        raw_wers.append(raw_wer)
        norm_wers.append(norm_wer)
        latencies.append(pred_info["latency_ms"])

        if cat not in category_metrics:
            category_metrics[cat] = {"raw_cer": [], "norm_cer": [], "norm_wer": [], "latency": []}
        category_metrics[cat]["raw_cer"].append(raw_cer)
        category_metrics[cat]["norm_cer"].append(norm_cer)
        category_metrics[cat]["norm_wer"].append(norm_wer)
        category_metrics[cat]["latency"].append(pred_info["latency_ms"])

        print(f"{s['id']:<6} | {cat:<10} | {raw_cer:7.2%} | {norm_cer:8.2%} | {pred_info['latency_ms']:5.1f}ms | \"{hyp_text}\"")

        results.append({
            "id": s["id"],
            "image": s["image"],
            "category": cat,
            "ground_truth": ref_text,
            "prediction": hyp_text,
            "raw_cer": float(raw_cer),
            "raw_wer": float(raw_wer),
            "norm_cer": float(norm_cer),
            "norm_wer": float(norm_wer),
            "exact_match": bool(is_exact),
            "norm_exact_match": bool(is_norm_exact),
            "latency_ms": float(pred_info["latency_ms"]),
            "chars_per_sec": float(pred_info["chars_per_sec"])
        })

    print("=" * 95)
    mean_raw_cer = float(np.mean(raw_cers))
    mean_norm_cer = float(np.mean(norm_cers))
    mean_raw_wer = float(np.mean(raw_wers))
    mean_norm_wer = float(np.mean(norm_wers))
    mean_lat = float(np.mean(latencies))
    em_acc = (exact_matches / len(samples)) * 100.0
    norm_em_acc = (norm_exact_matches / len(samples)) * 100.0

    print(f"Raw Exact Match (EM)   : {em_acc:.1f}% ({exact_matches}/{len(samples)})")
    print(f"Norm Exact Match (NEM) : {norm_em_acc:.1f}% ({norm_exact_matches}/{len(samples)})")
    print(f"Mean Normalized CER    : {mean_norm_cer:.2%} (Character Accuracy: {100.0 - mean_norm_cer*100:.2f}%)")
    print(f"Mean Normalized WER    : {mean_norm_wer:.2%}")
    print(f"Average OCR Latency    : {mean_lat:.1f} ms")

    summary = {
        "model": "microsoft/trocr-small-printed",
        "num_samples": len(samples),
        "exact_match_accuracy": em_acc,
        "normalized_exact_match": norm_em_acc,
        "mean_raw_cer": mean_raw_cer,
        "mean_norm_cer": mean_norm_cer,
        "mean_raw_wer": mean_raw_wer,
        "mean_norm_wer": mean_norm_wer,
        "average_latency_ms": mean_lat,
        "category_metrics": {
            cat: {
                "raw_cer": float(np.mean(m["raw_cer"])),
                "norm_cer": float(np.mean(m["norm_cer"])),
                "norm_wer": float(np.mean(m["norm_wer"])),
                "mean_latency_ms": float(np.mean(m["latency"]))
            }
            for cat, m in category_metrics.items()
        },
        "results": results
    }
    return summary


def plot_diagnostics(summary: Dict[str, Any], output_path: str):
    """Generates a 4-panel diagnostic figure visualizing TrOCR performance."""
    results = summary["results"]
    cat_metrics = summary["category_metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # -------------------------------------------------------------
    # Panel 1: OCR Gallery - Input Image vs Recognized Text
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_title("TrOCR Recognition Gallery & Visual Verification", fontsize=13, fontweight="bold", pad=10)
    ax1.axis("off")

    canvas = np.ones((520, 520, 3), dtype=np.uint8) * 248
    coords = [
        (10, 10), (10, 140), (10, 270), (10, 400)
    ]
    disp_samples = results[:4]
    for idx, item in enumerate(disp_samples):
        img_p = os.path.join(IMAGES_DIR, item["image"])
        raw_img = Image.open(img_p).convert("RGB").resize((220, 45))
        img_arr = np.array(raw_img)
        x, y = coords[idx]
        canvas[y:y+45, x:x+220] = img_arr

        # Annotate text
        color = "#2E7D32" if item["norm_exact_match"] else ("#F57C00" if item["norm_cer"] < 0.1 else "#D32F2F")
        badge = "100% MATCH" if item["norm_exact_match"] else f"CER: {item['norm_cer']:.1%}"
        ax1.text(x + 235, y + 16, f"Ref: \"{item['ground_truth'][:32]}\"", fontsize=8.5, fontweight="bold", color="#37474F")
        ax1.text(x + 235, y + 34, f"OCR: \"{item['prediction'][:32]}\"", fontsize=8.5, fontweight="bold", color=color)
        ax1.text(x + 235, y + 50, f"Status: {badge} | {item['latency_ms']:.1f}ms", fontsize=7.5, color="#616161")

    ax1.imshow(canvas)
    ax1.set_xlim(0, 520)
    ax1.set_ylim(520, 0)

    # -------------------------------------------------------------
    # Panel 2: CER & WER Across Categories
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    cat_names = list(cat_metrics.keys())
    cer_vals = [cat_metrics[c]["norm_cer"] * 100.0 for c in cat_names]
    wer_vals = [cat_metrics[c]["norm_wer"] * 100.0 for c in cat_names]

    x_idx = np.arange(len(cat_names))
    bar_w = 0.35

    ax2.bar(x_idx - bar_w/2, cer_vals, width=bar_w, label="CER (%)", color="#1976D2", edgecolor="#0D47A1", alpha=0.9)
    ax2.bar(x_idx + bar_w/2, wer_vals, width=bar_w, label="WER (%)", color="#FF9800", edgecolor="#E65100", alpha=0.9)

    ax2.set_title("Character (CER) and Word (WER) Error Rates", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(cat_names, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Error Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(max(cer_vals + wer_vals, default=10), 10) + 15)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=10)

    # -------------------------------------------------------------
    # Panel 3: Latency vs. Text Length
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    lengths = [len(r["prediction"]) for r in results]
    lats = [r["latency_ms"] for r in results]

    ax3.scatter(lengths, lats, color="#00897B", s=110, edgecolor="#004D40", zorder=3)
    if len(lengths) > 1:
        p = np.polyfit(lengths, lats, 1)
        fit_x = np.linspace(min(lengths), max(lengths), 50)
        fit_y = np.polyval(p, fit_x)
        ax3.plot(fit_x, fit_y, color="#E91E63", linestyle="--", label=f"Trend: {p[0]:.1f} ms/char")

    ax3.set_title("Inference Latency vs. Character Length", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Recognized String Length (characters)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=10)

    # -------------------------------------------------------------
    # Panel 4: Architectural Specs & Summary Card
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("TrOCR Architecture & Benchmark Summary", fontsize=13, fontweight="bold", pad=10)

    info_text = (
        "Model: microsoft/trocr-small-printed\n"
        "---------------------------------------------------\n"
        "• Vision Encoder  : Data-efficient Image Transformer (DeiT)\n"
        "• Text Decoder    : Autoregressive MiniLM/RoBERTa Head\n"
        "• Total Parameters: 61,596,672 (61.6M)\n"
        "• Pretraining Data: Large-scale synthetic printed text\n\n"
        "Benchmark Performance:\n"
        "---------------------------------------------------\n"
        f"• Test Samples       : {summary['num_samples']} document patches\n"
        f"• Normalized EM (NEM): {summary['normalized_exact_match']:.1f}%\n"
        f"• Raw Exact Match    : {summary['exact_match_accuracy']:.1f}%\n"
        f"• Mean Normalized CER: {summary['mean_norm_cer']:.2%}\n"
        f"• Character Accuracy : {100.0 - summary['mean_norm_cer']*100:.2f}%\n"
        f"• Mean Normalized WER: {summary['mean_norm_wer']:.2%}\n"
        f"• Average Latency    : {summary['average_latency_ms']:.1f} ms / image\n"
        f"• Hardware Platform  : CPU Inference\n\n"
        "Key Advantages:\n"
        "• End-to-end OCR without separate character segmentation\n"
        "• Robust to scanning artifacts, fonts, and degraded noise\n"
        "• Compact footprint suitable for edge & CPU deployment"
    )

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=10.5, fontfamily="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#ECEFF1", edgecolor="#90A4AE", alpha=0.9))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic report successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Optical Character Recognition with Transformers (TrOCR)")
    parser.add_argument("--image", type=str, default=None, help="Path to image file for OCR")
    parser.add_argument("--model", type=str, default="microsoft/trocr-small-printed", help="TrOCR checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive OCR console")
    args = parser.parse_args()

    engine = TrOCREngine(model_id=args.model)

    # Single-image query mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image '{args.image}' not found.")
            sys.exit(1)
        raw_img = Image.open(args.image).convert("RGB")
        res = engine.recognize_text(raw_img)
        print("\n" + "=" * 50)
        print(f"Image       : {args.image}")
        print(f"Recognized  : \"{res['text']}\"")
        print(f"Latency     : {res['latency_ms']:.1f} ms ({res['chars_per_sec']:.1f} chars/sec)")
        print("=" * 50)
        return

    # Interactive mode
    if args.interactive:
        print("\n--- Interactive TrOCR Console (Type 'exit' to quit) ---")
        while True:
            img_in = input("\nEnter image path (or Enter for 'sample_quote.png'): ").strip()
            if img_in.lower() == "exit":
                break
            if not img_in:
                img_in = os.path.join(IMAGES_DIR, "sample_quote.png")
            if not os.path.exists(img_in):
                print(f"File '{img_in}' does not exist.")
                continue

            raw_img = Image.open(img_in).convert("RGB")
            res = engine.recognize_text(raw_img)
            print(f">> TrOCR Output: \"{res['text']}\" ({res['latency_ms']:.1f}ms)")
        return

    # Benchmark evaluation mode
    dataset_path = os.path.join(DATA_DIR, "ocr_test_dataset.json")
    summary = evaluate_benchmark(engine, dataset_path)

    # Save JSON results
    json_path = os.path.join(OUTPUTS_DIR, "trocr_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation results saved to: {json_path}")

    # Generate diagnostic report
    fig_path = os.path.join(OUTPUTS_DIR, "trocr_diagnostic_report.png")
    plot_diagnostics(summary, fig_path)


if __name__ == "__main__":
    main()

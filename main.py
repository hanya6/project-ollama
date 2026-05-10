"""
KLASIFIKASI TEMA PELAYANAN KESEHATAN IBU DAN ANAK
PADA TEKS LAPORAN POSYANDU MENGGUNAKAN LARGE LANGUAGE MODELS (LLM)
DAN ZERO-SHOT LEARNING

Alur:
  1. Data tabular (CSV) -> Generate teks laporan naratif
  2. Teks laporan -> Klasifikasi tema via LLM Zero-Shot
  3. Evaluasi hasil klasifikasi

Penggunaan:
  python main.py                    # Pipeline lengkap
  python main.py --generate         # Generate teks saja
  python main.py --classify         # Klasifikasi saja
  python main.py --model mistral    # Model tertentu
  python main.py --reasoning        # Dengan Chain-of-Thought
  python main.py --limit 30         # Batasi data
  python main.py --single "teks"    # Satu teks saja
"""

import argparse
import os
import sys
import time

import pandas as pd

from config import (
    OLLAMA_MODEL, RESULTS_DIR, OUTPUT_CSV,
    GENERATED_DATA_PATH, TEMA_LABELS, VERBOSE,
)
from src.text_generator import TextGenerator
from src.classifier import OllamaClassifier
from src.data_loader import load_generated_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Klasifikasi Tema Pelayanan KIA - LLM Zero-Shot Learning"
    )
    parser.add_argument("--model", "-m", default=OLLAMA_MODEL, help="Model Ollama")
    parser.add_argument("--generate", "-g", action="store_true", help="Generate teks saja")
    parser.add_argument("--classify", "-c", action="store_true", help="Klasifikasi saja")
    parser.add_argument("--single", "-s", type=str, help="Klasifikasi satu teks")
    parser.add_argument("--reasoning", "-r", action="store_true", help="Chain-of-Thought")
    parser.add_argument("--output", "-o", default=OUTPUT_CSV, help="Output CSV")
    parser.add_argument("--delay", "-d", type=float, default=0.5, help="Delay antar request")
    parser.add_argument("--limit", "-l", type=int, help="Batasi jumlah data")
    parser.add_argument("--n-per-tema", type=int, help="Data per tema saat generate")
    return parser.parse_args()


def print_banner():
    print("""
================================================================
  KLASIFIKASI TEMA PELAYANAN KESEHATAN IBU DAN ANAK
  PADA TEKS LAPORAN POSYANDU
  MENGGUNAKAN LARGE LANGUAGE MODELS (LLM) DAN ZERO-SHOT LEARNING
================================================================
""")


def step_generate(args) -> pd.DataFrame:
    """Step 1: Generate teks laporan dari data CSV."""
    print("=" * 60)
    print("STEP 1: GENERATE TEKS LAPORAN DARI DATA TABULAR")
    print("=" * 60)

    gen = TextGenerator(seed=42)
    df = gen.generate_and_save(n_per_tema=args.n_per_tema)

    print("\nContoh teks per tema:")
    for tema in TEMA_LABELS:
        sample = df[df["tema_aktual"] == tema].iloc[0]
        print(f"\n  [{tema}]")
        print(f"  {sample['teks_laporan'][:150]}...")

    return df


def step_classify(args, df=None) -> pd.DataFrame:
    """Step 2: Klasifikasi tema dengan LLM Zero-Shot."""
    print("\n" + "=" * 60)
    print("STEP 2: KLASIFIKASI ZERO-SHOT DENGAN LLM")
    print("=" * 60)

    if df is None:
        df = load_generated_data()

    if args.limit:
        df = df.head(args.limit)
        print(f"[INFO] Data dibatasi: {args.limit} baris")

    print(f"[INFO] Total: {len(df)} teks | Model: {args.model} | Reasoning: {args.reasoning}")

    classifier = OllamaClassifier(model=args.model)

    if not classifier.check_connection():
        print("\n[ERROR] Gagal terhubung ke Ollama!")
        print("  1. Install: curl -fsSL https://ollama.com/install.sh | sh")
        print("  2. Jalankan: ollama serve")
        print(f"  3. Pull model: ollama pull {args.model}")
        sys.exit(1)

    start = time.time()
    df_result = classifier.classify_dataframe(df, with_reasoning=args.reasoning, delay=args.delay)
    total_time = time.time() - start

    # Ringkasan
    print(f"\n{'='*60}")
    print("RINGKASAN HASIL")
    print(f"{'='*60}")
    print(f"  Total: {len(df_result)} | Waktu: {total_time:.1f}s | Avg: {total_time/len(df_result):.2f}s/teks")

    print(f"\n  Distribusi prediksi:")
    for tema, cnt in df_result["tema_prediksi"].value_counts().items():
        print(f"    - {tema}: {cnt} ({cnt/len(df_result)*100:.1f}%)")

    if "tema_aktual" in df_result.columns:
        correct = (df_result["tema_prediksi"] == df_result["tema_aktual"]).sum()
        print(f"\n  Akurasi: {correct}/{len(df_result)} ({correct/len(df_result)*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_result.to_csv(args.output, index=False)
    print(f"\n[INFO] Hasil disimpan: {args.output}")

    return df_result


def main():
    print_banner()
    args = parse_args()

    # Mode single text
    if args.single:
        classifier = OllamaClassifier(model=args.model)
        if not classifier.check_connection():
            sys.exit(1)
        result = classifier.classify_single(args.single, with_reasoning=args.reasoning)
        print(f"\n  Teks  : {args.single[:100]}...")
        print(f"  Tema  : {result['tema_prediksi']}")
        print(f"  Waktu : {result['waktu_proses']}s")
        if args.reasoning and result["alasan"]:
            print(f"  Alasan: {result['alasan']}")
        return

    # Mode generate only
    if args.generate:
        step_generate(args)
        print(f"\n[INFO] Selesai! File: {GENERATED_DATA_PATH}")
        print("[INFO] Selanjutnya: python main.py --classify")
        return

    # Mode classify only
    if args.classify:
        step_classify(args)
        print("\n[INFO] Selanjutnya: python evaluate.py")
        return

    # Pipeline lengkap
    print("[INFO] PIPELINE LENGKAP: Generate -> Klasifikasi\n")
    df = step_generate(args)
    step_classify(args, df)
    print(f"\n[INFO] Selesai! Jalankan 'python evaluate.py' untuk evaluasi lengkap.")


if __name__ == "__main__":
    main()

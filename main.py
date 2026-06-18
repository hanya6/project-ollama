"""
KLASIFIKASI TEMA PELAYANAN KESEHATAN IBU DAN ANAK
PADA TEKS LAPORAN POSYANDU MENGGUNAKAN LARGE LANGUAGE MODELS (LLM)
DAN ZERO-SHOT LEARNING
"""

import argparse
import os
import sys
import time

import pandas as pd

from config import (
    OLLAMA_MODEL,
    OUTPUT_CSV,
    GENERATED_DATA_PATH,
    TEMA_LABELS,
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
    print("=" * 60)
    print("STEP 1: GENERATE TEKS LAPORAN DARI DATA TABULAR")
    print("=" * 60)

    gen = TextGenerator(seed=42)

    print("\n[INFO] Kolom pada file gizi_balita.csv:")
    print(gen.gizi_df.columns.tolist())

    df = gen.generate_and_save(n_per_tema=args.n_per_tema)

    print("\nContoh teks per tema:")
    for tema in TEMA_LABELS:
        subset = df[df["tema_aktual"] == tema]
        if len(subset) > 0:
            sample = subset.iloc[0]
            print(f"\n  [{tema}]")
            print(f"  {sample['teks_laporan'][:150]}...")

    return df


def step_classify(args, df=None) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2: KLASIFIKASI ZERO-SHOT DENGAN LLM")
    print("=" * 60)

    if df is None:
        df = load_generated_data()

    if args.limit:
        df = df.head(args.limit)
        print(f"[INFO] Data dibatasi: {args.limit} baris")

    print(f"[INFO] Total: {len(df)} teks")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Reasoning: {args.reasoning}")

    classifier = OllamaClassifier(model=args.model)

    if not classifier.check_connection():
        print("\n[ERROR] Gagal terhubung ke Ollama!")
        print("1. Jalankan: ollama serve")
        print(f"2. Pull model: ollama pull {args.model}")
        sys.exit(1)

    start = time.time()

    df_result = classifier.classify_dataframe(
        df,
        with_reasoning=args.reasoning,
        delay=args.delay
    )

    total_time = time.time() - start

    print(f"\n{'=' * 60}")
    print("RINGKASAN HASIL")
    print(f"{'=' * 60}")

    print(f"Total data : {len(df_result)}")
    print(f"Waktu      : {total_time:.1f} detik")
    print(f"Rata-rata  : {total_time / len(df_result):.2f} detik/teks")

    print("\nDistribusi prediksi:")
    for tema, cnt in df_result["tema_prediksi"].value_counts().items():
        print(f"- {tema}: {cnt} ({cnt / len(df_result) * 100:.1f}%)")

    if "tema_aktual" in df_result.columns:
        correct = (df_result["tema_prediksi"] == df_result["tema_aktual"]).sum()
        acc = correct / len(df_result) * 100
        print(f"\nAkurasi: {correct}/{len(df_result)} ({acc:.1f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_result.to_csv(args.output, index=False)

    print(f"\n[INFO] Hasil disimpan: {args.output}")

    return df_result


def main():
    print_banner()
    args = parse_args()

    if args.single:
        classifier = OllamaClassifier(model=args.model)

        if not classifier.check_connection():
            print("[ERROR] Ollama belum terhubung.")
            sys.exit(1)

        result = classifier.classify_single(
            args.single,
            with_reasoning=args.reasoning
        )

        print(f"\nTeks  : {args.single[:100]}...")
        print(f"Tema  : {result['tema_prediksi']}")
        print(f"Waktu : {result['waktu_proses']}s")

        if args.reasoning and result.get("alasan"):
            print(f"Alasan: {result['alasan']}")

        return

    if args.generate:
        step_generate(args)
        print(f"\n[INFO] Selesai generate. File: {GENERATED_DATA_PATH}")
        print("[INFO] Selanjutnya jalankan: python main.py --classify")
        return

    if args.classify:
        step_classify(args)
        print("\n[INFO] Selanjutnya jalankan: python evaluate.py")
        return

    print("[INFO] PIPELINE LENGKAP: Generate -> Klasifikasi\n")

    df = step_generate(args)
    step_classify(args, df)

    print("\n[INFO] Pipeline selesai.")
    print("[INFO] Jalankan evaluate.py untuk evaluasi lengkap.")


if __name__ == "__main__":
    main()

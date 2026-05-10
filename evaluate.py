"""
Evaluasi Hasil Klasifikasi Tema Pelayanan KIA
pada Teks Laporan Posyandu (Zero-Shot Learning + LLM)

Penggunaan:
  python evaluate.py
  python evaluate.py --input results/hasil_klasifikasi.csv
  python evaluate.py --model llama3
"""

import argparse
import os
import sys

import pandas as pd

from config import OUTPUT_CSV, RESULTS_DIR, OLLAMA_MODEL
from src.evaluator import ClassificationEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Klasifikasi Tema Posyandu")
    parser.add_argument("--input", "-i", default=OUTPUT_CSV, help="File hasil klasifikasi")
    parser.add_argument("--model", "-m", default=OLLAMA_MODEL, help="Nama model")
    args = parser.parse_args()

    print("""
================================================================
  EVALUASI KLASIFIKASI TEMA PELAYANAN KIA
  PADA TEKS LAPORAN POSYANDU
  Metode: Zero-Shot Learning + LLM
================================================================
""")

    if not os.path.exists(args.input):
        print(f"[ERROR] File tidak ditemukan: {args.input}")
        print("[INFO] Jalankan 'python main.py' terlebih dahulu.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"[INFO] Data: {len(df)} baris dari {args.input}")

    for col in ["tema_aktual", "tema_prediksi"]:
        if col not in df.columns:
            print(f"[ERROR] Kolom '{col}' tidak ditemukan!")
            sys.exit(1)

    y_true = df["tema_aktual"].tolist()
    y_pred = df["tema_prediksi"].tolist()

    evaluator = ClassificationEvaluator()
    evaluator.evaluate_all(y_true, y_pred, model_name=args.model)

    print("\n[INFO] File output:")
    files = [
        ("Laporan", os.path.join(RESULTS_DIR, "laporan_evaluasi.txt")),
        ("Confusion Matrix", os.path.join(RESULTS_DIR, "confusion_matrix.png")),
        ("Grafik Metrik", os.path.join(RESULTS_DIR, "classification_report.png")),
    ]
    for name, path in files:
        status = "OK" if os.path.exists(path) else "-"
        print(f"  [{status}] {name}: {path}")

    print("\n[INFO] Evaluasi selesai!")


if __name__ == "__main__":
    main()

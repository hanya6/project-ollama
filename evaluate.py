"""
Script Evaluasi & Visualisasi Hasil Klasifikasi
Membaca hasil klasifikasi dan menghasilkan laporan evaluasi lengkap.

Penggunaan:
    python evaluate.py                                         # Evaluasi hasil default
    python evaluate.py --input results/hasil_klasifikasi.csv   # File tertentu
    python evaluate.py --model llama3                          # Nama model untuk laporan
"""

import argparse
import os
import sys

import pandas as pd

from config import OUTPUT_CSV, RESULTS_DIR, OLLAMA_MODEL
from src.evaluator import ClassificationEvaluator


def parse_arguments():
    """Parse argumen command line."""
    parser = argparse.ArgumentParser(
        description="Evaluasi Hasil Klasifikasi Tema Pelayanan KIA pada Laporan Posyandu"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=OUTPUT_CSV,
        help=f"Path file hasil klasifikasi CSV (default: {OUTPUT_CSV})"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Nama model untuk laporan (default: {OLLAMA_MODEL})"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=RESULTS_DIR,
        help=f"Direktori output (default: {RESULTS_DIR})"
    )

    return parser.parse_args()


def main():
    """Fungsi utama evaluasi."""
    print("""
+======================================================================+
|  EVALUASI KLASIFIKASI TEMA PELAYANAN KESEHATAN IBU DAN ANAK          |
|  PADA TEKS LAPORAN POSYANDU                                          |
|  Metode: Zero-Shot Learning dengan Large Language Model (LLM)        |
+======================================================================+
    """)

    args = parse_arguments()

    # Cek file input
    if not os.path.exists(args.input):
        print(f"[ERROR] File hasil klasifikasi tidak ditemukan: {args.input}")
        print("[INFO] Jalankan 'python main.py' terlebih dahulu untuk melakukan klasifikasi.")
        sys.exit(1)

    # Load hasil klasifikasi
    print(f"[INFO] Memuat hasil klasifikasi: {args.input}")
    df = pd.read_csv(args.input)

    print(f"[INFO] Total data: {len(df)} baris")

    # Cek kolom yang diperlukan
    if "tema_aktual" not in df.columns:
        print("[ERROR] Kolom 'tema_aktual' tidak ditemukan!")
        print("[INFO] Evaluasi memerlukan data dengan label aktual.")
        sys.exit(1)

    if "tema_prediksi" not in df.columns:
        print("[ERROR] Kolom 'tema_prediksi' tidak ditemukan!")
        print("[INFO] Pastikan file CSV adalah output dari main.py.")
        sys.exit(1)

    # Ambil label
    y_true = df["tema_aktual"].tolist()
    y_pred = df["tema_prediksi"].tolist()

    print(f"[INFO] Label aktual unik: {len(set(y_true))} - {set(y_true)}")
    print(f"[INFO] Label prediksi unik: {len(set(y_pred))} - {set(y_pred)}")

    # Inisialisasi evaluator
    evaluator = ClassificationEvaluator()

    # Jalankan evaluasi lengkap
    metrics = evaluator.evaluate_all(
        y_true=y_true,
        y_pred=y_pred,
        model_name=args.model
    )

    # Ringkasan file output
    print("\n" + "=" * 60)
    print("FILE OUTPUT EVALUASI:")
    print("=" * 60)

    output_files = [
        ("Laporan Evaluasi", os.path.join(RESULTS_DIR, "laporan_evaluasi.txt")),
        ("Confusion Matrix", os.path.join(RESULTS_DIR, "confusion_matrix.png")),
        ("Grafik Metrik", os.path.join(RESULTS_DIR, "classification_report.png")),
        ("Distribusi Prediksi", os.path.join(RESULTS_DIR, "distribusi_prediksi.png")),
    ]

    for name, path in output_files:
        exists = "OK" if os.path.exists(path) else "GAGAL"
        print(f"  [{exists}] {name}: {path}")

    print("\n[INFO] Evaluasi selesai!")
    print("[INFO] Buka file gambar (.png) untuk melihat visualisasi.")


if __name__ == "__main__":
    main()

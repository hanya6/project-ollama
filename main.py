"""
Script Utama - Klasifikasi Tema Pelayanan Kesehatan Ibu dan Anak
pada Teks Laporan Posyandu menggunakan LLM dan Zero-Shot Learning

Penggunaan:
    python main.py                          # Klasifikasi data sampel
    python main.py --model mistral          # Gunakan model tertentu
    python main.py --reasoning              # Dengan Chain-of-Thought
    python main.py --input data/custom.csv  # Data kustom
    python main.py --single "teks laporan"  # Klasifikasi satu teks
"""

import argparse
import os
import sys
import time

import pandas as pd

from config import (
    OLLAMA_MODEL,
    RESULTS_DIR,
    OUTPUT_CSV,
    TEMA_LABELS,
    VERBOSE,
)
from src.classifier import OllamaClassifier
from src.data_loader import load_sample_data, load_custom_data


def parse_arguments():
    """Parse argumen command line."""
    parser = argparse.ArgumentParser(
        description="Klasifikasi Tema Laporan Posyandu menggunakan LLM Zero-Shot Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py                              # Klasifikasi data sampel
  python main.py --model llama3               # Gunakan model llama3
  python main.py --model mistral --reasoning  # Mistral + reasoning
  python main.py --input data/custom.csv      # Data kustom
  python main.py --single "Hari ini dilaksanakan penimbangan 50 balita"
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Model Ollama yang digunakan (default: {OLLAMA_MODEL})"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path ke file CSV input (default: data sampel)"
    )
    
    parser.add_argument(
        "--single", "-s",
        type=str,
        default=None,
        help="Klasifikasi satu teks laporan secara langsung"
    )
    
    parser.add_argument(
        "--reasoning", "-r",
        action="store_true",
        help="Aktifkan mode Chain-of-Thought reasoning"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=OUTPUT_CSV,
        help=f"Path file output CSV (default: {OUTPUT_CSV})"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.5,
        help="Jeda antar request dalam detik (default: 0.5)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Batasi jumlah data yang diproses"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=VERBOSE,
        help="Tampilkan detail proses"
    )
    
    return parser.parse_args()


def print_banner():
    """Tampilkan banner aplikasi."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║  KLASIFIKASI TEMA PELAYANAN KESEHATAN IBU DAN ANAK             ║
║  pada Teks Laporan Posyandu                                    ║
║  menggunakan Large Language Models (LLM) & Zero-Shot Learning  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def classify_single_text(classifier, text, with_reasoning, verbose):
    """Klasifikasi satu teks dan tampilkan hasilnya."""
    print(f"\n{'='*60}")
    print("MODE: Klasifikasi Teks Tunggal")
    print(f"{'='*60}")
    print(f"\nTeks: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
    print(f"\nMemproses...")
    
    result = classifier.classify_single(text, with_reasoning=with_reasoning)
    
    print(f"\n{'─'*60}")
    print(f"  HASIL KLASIFIKASI")
    print(f"{'─'*60}")
    print(f"  Tema    : {result['tema_prediksi']}")
    print(f"  Waktu   : {result['waktu_proses']} detik")
    print(f"  Model   : {result['model']}")
    
    if with_reasoning and result.get("alasan"):
        print(f"  Alasan  : {result['alasan']}")
    
    if verbose:
        print(f"\n  Raw Response: {result['raw_response']}")
    
    print(f"{'─'*60}")
    
    return result


def classify_batch_data(classifier, args):
    """Klasifikasi batch data dari file CSV."""
    # Load data
    print(f"\n{'='*60}")
    print("MODE: Klasifikasi Batch Data")
    print(f"{'='*60}\n")
    
    if args.input:
        df = load_custom_data(args.input)
    else:
        df = load_sample_data()
    
    # Batasi data jika diperlukan
    if args.limit:
        df = df.head(args.limit)
        print(f"[INFO] Data dibatasi: {args.limit} baris pertama")
    
    print(f"\n[INFO] Total data: {len(df)} teks laporan")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Reasoning: {'Ya' if args.reasoning else 'Tidak'}")
    print(f"[INFO] Delay: {args.delay}s antar request")
    
    # Konfirmasi
    print(f"\n[?] Lanjutkan klasifikasi? (tekan Enter atau ketik 'y')")
    
    # Mulai klasifikasi
    start_time = time.time()
    
    df_result = classifier.classify_dataframe(
        df,
        text_column="teks_laporan",
        with_reasoning=args.reasoning,
        delay=args.delay
    )
    
    total_time = time.time() - start_time
    
    # Tampilkan ringkasan
    print(f"\n{'='*60}")
    print("RINGKASAN HASIL KLASIFIKASI")
    print(f"{'='*60}")
    print(f"  Total teks        : {len(df_result)}")
    print(f"  Total waktu       : {total_time:.1f} detik")
    print(f"  Rata-rata/teks    : {total_time/len(df_result):.2f} detik")
    
    # Distribusi hasil prediksi
    print(f"\n  Distribusi Prediksi:")
    for tema, count in df_result["tema_prediksi"].value_counts().items():
        pct = count / len(df_result) * 100
        print(f"    - {tema}: {count} ({pct:.1f}%)")
    
    # Hitung akurasi jika ada label aktual
    if "tema_aktual" in df_result.columns:
        correct = (df_result["tema_prediksi"] == df_result["tema_aktual"]).sum()
        accuracy = correct / len(df_result) * 100
        print(f"\n  Akurasi           : {correct}/{len(df_result)} ({accuracy:.1f}%)")
        
        # Tampilkan yang salah
        wrong = df_result[df_result["tema_prediksi"] != df_result["tema_aktual"]]
        if len(wrong) > 0:
            print(f"\n  Prediksi Salah ({len(wrong)} teks):")
            for _, row in wrong.iterrows():
                print(f"    ID {row['id']}: "
                      f"Aktual='{row['tema_aktual']}' | "
                      f"Prediksi='{row['tema_prediksi']}'")
    
    # Simpan hasil
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_result.to_csv(args.output, index=False)
    print(f"\n[INFO] Hasil disimpan ke: {args.output}")
    
    # Simpan detail jika verbose
    if args.verbose:
        detail_path = args.output.replace(".csv", "_detail.csv")
        df_result.to_csv(detail_path, index=False)
        print(f"[INFO] Detail disimpan ke: {detail_path}")
    
    print(f"{'='*60}\n")
    
    return df_result


def main():
    """Fungsi utama."""
    print_banner()
    
    # Parse argumen
    args = parse_arguments()
    
    # Inisialisasi classifier
    print(f"[INFO] Inisialisasi classifier...")
    print(f"[INFO] Model: {args.model}")
    
    classifier = OllamaClassifier(model=args.model)
    
    # Cek koneksi ke Ollama
    print(f"[INFO] Mengecek koneksi ke Ollama...")
    if not classifier.check_connection():
        print("\n[ERROR] Gagal terhubung ke Ollama!")
        print("[INFO] Pastikan:")
        print("       1. Ollama sudah terinstall (curl -fsSL https://ollama.com/install.sh | sh)")
        print("       2. Ollama server berjalan (ollama serve)")
        print(f"       3. Model tersedia (ollama pull {args.model})")
        sys.exit(1)
    
    print("[INFO] Koneksi berhasil!\n")
    
    # Mode klasifikasi
    if args.single:
        # Mode single text
        result = classify_single_text(
            classifier, args.single, args.reasoning, args.verbose
        )
    else:
        # Mode batch
        df_result = classify_batch_data(classifier, args)
    
    print("[INFO] Selesai! Gunakan 'python evaluate.py' untuk evaluasi detail.")


if __name__ == "__main__":
    main()

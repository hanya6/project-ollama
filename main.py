"""
Script Utama - Klasifikasi Tema Pelayanan Kesehatan Ibu dan Anak
pada Teks Laporan Posyandu menggunakan LLM dan Zero-Shot Learning

Alur kerja:
1. Membaca data tabular dari 3 file CSV (gizi, imunisasi, kesehatan)
2. Generate teks laporan naratif dari data tabular
3. Klasifikasi tema menggunakan LLM (Ollama) dengan Zero-Shot Learning
4. Evaluasi hasil klasifikasi

Penggunaan:
    python main.py                          # Jalankan pipeline lengkap
    python main.py --generate               # Generate teks saja
    python main.py --classify               # Klasifikasi dari data generated
    python main.py --model mistral          # Gunakan model tertentu
    python main.py --reasoning              # Dengan Chain-of-Thought
    python main.py --limit 30              # Batasi jumlah data
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
    GENERATED_DATA_PATH,
    TEMA_LABELS,
    VERBOSE,
)
from src.text_generator import TextGenerator
from src.classifier import OllamaClassifier
from src.data_loader import load_generated_data


def parse_arguments():
    """Parse argumen command line."""
    parser = argparse.ArgumentParser(
        description="Klasifikasi Tema Pelayanan KIA pada Laporan Posyandu (LLM + Zero-Shot Learning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py                              # Pipeline lengkap (generate + klasifikasi)
  python main.py --generate                   # Generate teks laporan saja
  python main.py --classify                   # Klasifikasi dari generated data
  python main.py --model llama3 --reasoning   # Llama3 + Chain-of-Thought
  python main.py --limit 50                   # Batasi 50 data
  python main.py --single "Hari ini dilakukan penimbangan balita..."
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Model Ollama yang digunakan (default: {OLLAMA_MODEL})"
    )

    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Hanya generate teks laporan dari data CSV (tanpa klasifikasi)"
    )

    parser.add_argument(
        "--classify", "-c",
        action="store_true",
        help="Hanya klasifikasi dari data yang sudah di-generate"
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
        "--english",
        action="store_true",
        help="Gunakan prompt bahasa Inggris"
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
        "--n-per-tema",
        type=int,
        default=None,
        help="Jumlah data per tema saat generate (default: semua)"
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
+======================================================================+
|  KLASIFIKASI TEMA PELAYANAN KESEHATAN IBU DAN ANAK                   |
|  PADA TEKS LAPORAN POSYANDU                                          |
|  MENGGUNAKAN LARGE LANGUAGE MODELS (LLM) DAN ZERO-SHOT LEARNING     |
+======================================================================+
    """
    print(banner)


def step_generate(args) -> pd.DataFrame:
    """
    STEP 1: Generate teks laporan naratif dari data tabular CSV.

    Args:
        args: Argumen command line

    Returns:
        DataFrame berisi teks laporan dan label tema
    """
    print("\n" + "=" * 60)
    print("STEP 1: GENERATE TEKS LAPORAN DARI DATA TABULAR")
    print("=" * 60)

    print("\n[INFO] Mengubah data tabular CSV menjadi teks laporan naratif...")
    print("[INFO] Sumber data:")
    print("       - data/gizi_balita.csv")
    print("       - data/imunisasi_bayi.csv")
    print("       - data/kesehatan_balita.csv")

    generator = TextGenerator(seed=42)
    df = generator.generate_and_save(
        n_per_tema=args.n_per_tema,
        shuffle=True
    )

    # Tampilkan contoh
    print(f"\n[INFO] Contoh teks yang dihasilkan:")
    for tema in TEMA_LABELS:
        sample = df[df["tema_aktual"] == tema].iloc[0]
        print(f"\n  Tema: {tema}")
        print(f"  Teks: \"{sample['teks_laporan'][:150]}...\"")

    return df


def step_classify(args, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    STEP 2: Klasifikasi tema menggunakan LLM Zero-Shot Learning.

    Args:
        args: Argumen command line
        df: DataFrame teks laporan (opsional, akan load dari file jika None)

    Returns:
        DataFrame dengan hasil klasifikasi
    """
    print("\n" + "=" * 60)
    print("STEP 2: KLASIFIKASI TEMA DENGAN LLM (ZERO-SHOT LEARNING)")
    print("=" * 60)

    # Load data jika belum ada
    if df is None:
        df = load_generated_data()

    # Batasi data jika diperlukan
    if args.limit:
        df = df.head(args.limit)
        print(f"[INFO] Data dibatasi: {args.limit} baris")

    print(f"\n[INFO] Total data: {len(df)} teks laporan")
    print(f"[INFO] Model LLM: {args.model}")
    print(f"[INFO] Mode: {'Reasoning' if args.reasoning else 'English' if args.english else 'Direct Zero-Shot'}")
    print(f"[INFO] Delay: {args.delay}s antar request")

    # Inisialisasi classifier
    classifier = OllamaClassifier(model=args.model)

    # Cek koneksi ke Ollama
    print(f"\n[INFO] Mengecek koneksi ke Ollama...")
    if not classifier.check_connection():
        print("\n[ERROR] Gagal terhubung ke Ollama!")
        print("[INFO] Pastikan:")
        print("       1. Ollama sudah terinstall (curl -fsSL https://ollama.com/install.sh | sh)")
        print("       2. Ollama server berjalan (ollama serve)")
        print(f"       3. Model tersedia (ollama pull {args.model})")
        sys.exit(1)

    print("[INFO] Koneksi berhasil!")

    # Mulai klasifikasi
    start_time = time.time()

    df_result = classifier.classify_dataframe(
        df,
        text_column="teks_laporan",
        with_reasoning=args.reasoning,
        use_english=args.english,
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

    # Hitung akurasi
    if "tema_aktual" in df_result.columns:
        correct = (df_result["tema_prediksi"] == df_result["tema_aktual"]).sum()
        accuracy = correct / len(df_result) * 100
        print(f"\n  Akurasi           : {correct}/{len(df_result)} ({accuracy:.1f}%)")

        # Tampilkan beberapa yang salah
        wrong = df_result[df_result["tema_prediksi"] != df_result["tema_aktual"]]
        if len(wrong) > 0 and len(wrong) <= 20:
            print(f"\n  Prediksi Salah ({len(wrong)} teks):")
            for _, row in wrong.head(10).iterrows():
                print(f"    ID {row['id']}: "
                      f"Aktual='{row['tema_aktual']}' | "
                      f"Prediksi='{row['tema_prediksi']}'")
            if len(wrong) > 10:
                print(f"    ... dan {len(wrong) - 10} kesalahan lainnya")

    # Simpan hasil
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_result.to_csv(args.output, index=False)
    print(f"\n[INFO] Hasil klasifikasi disimpan ke: {args.output}")

    print(f"{'='*60}\n")

    return df_result


def classify_single_text(classifier, text, args):
    """Klasifikasi satu teks dan tampilkan hasilnya."""
    print(f"\n{'='*60}")
    print("MODE: Klasifikasi Teks Tunggal (Zero-Shot)")
    print(f"{'='*60}")
    print(f"\nTeks: \"{text[:150]}{'...' if len(text) > 150 else ''}\"")
    print(f"\nMemproses dengan model {args.model}...")

    result = classifier.classify_single(
        text,
        with_reasoning=args.reasoning,
        use_english=args.english
    )

    print(f"\n{'─'*60}")
    print(f"  HASIL KLASIFIKASI (ZERO-SHOT LEARNING)")
    print(f"{'─'*60}")
    print(f"  Tema    : {result['tema_prediksi']}")
    print(f"  Waktu   : {result['waktu_proses']} detik")
    print(f"  Model   : {result['model']}")

    if args.reasoning and result.get("alasan"):
        print(f"  Alasan  : {result['alasan']}")

    if args.verbose:
        print(f"\n  Raw Response: {result['raw_response']}")

    print(f"{'─'*60}")

    return result


def main():
    """Fungsi utama - menjalankan pipeline klasifikasi."""
    print_banner()

    # Parse argumen
    args = parse_arguments()

    # === MODE: Single Text ===
    if args.single:
        classifier = OllamaClassifier(model=args.model)
        print(f"[INFO] Mengecek koneksi ke Ollama...")
        if not classifier.check_connection():
            print("[ERROR] Gagal terhubung ke Ollama!")
            sys.exit(1)
        classify_single_text(classifier, args.single, args)
        return

    # === MODE: Generate Only ===
    if args.generate:
        step_generate(args)
        print("\n[INFO] Selesai! Teks laporan berhasil di-generate.")
        print(f"[INFO] File tersimpan di: {GENERATED_DATA_PATH}")
        print("[INFO] Jalankan 'python main.py --classify' untuk klasifikasi.")
        return

    # === MODE: Classify Only ===
    if args.classify:
        df_result = step_classify(args)
        print("[INFO] Selesai! Jalankan 'python evaluate.py' untuk evaluasi detail.")
        return

    # === MODE: Full Pipeline ===
    print("\n[INFO] Menjalankan PIPELINE LENGKAP:")
    print("       Step 1: Generate teks laporan dari data tabular")
    print("       Step 2: Klasifikasi tema dengan LLM Zero-Shot Learning")
    print("")

    # Step 1: Generate
    df_generated = step_generate(args)

    # Step 2: Classify
    df_result = step_classify(args, df_generated)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SELESAI!")
    print("=" * 60)
    print(f"\n[INFO] Output files:")
    print(f"       - Data generated : {GENERATED_DATA_PATH}")
    print(f"       - Hasil klasifikasi: {args.output}")
    print(f"\n[INFO] Langkah selanjutnya:")
    print(f"       Jalankan 'python evaluate.py' untuk evaluasi lengkap")
    print(f"       dengan confusion matrix, classification report, dan grafik.")


if __name__ == "__main__":
    main()

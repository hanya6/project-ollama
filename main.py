import argparse
import os
import sys
import time
from typing import Optional

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


# =====================================================
# ARGUMENT PARSER
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Klasifikasi Tema Pelayanan KIA menggunakan "
            "LLM dan Zero-Shot Learning"
        )
    )

    parser.add_argument(
        "--model",
        "-m",
        default=OLLAMA_MODEL,
        help="Nama model Ollama yang digunakan.",
    )

    parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Hanya melakukan generate teks laporan.",
    )

    parser.add_argument(
        "--classify",
        "-c",
        action="store_true",
        help="Hanya melakukan klasifikasi data yang sudah dibuat.",
    )

    parser.add_argument(
        "--single",
        "-s",
        type=str,
        help="Melakukan klasifikasi terhadap satu teks.",
    )

    parser.add_argument(
        "--reasoning",
        "-r",
        action="store_true",
        help="Meminta reasoning singkat dari model.",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=OUTPUT_CSV,
        help="Lokasi file output hasil klasifikasi.",
    )

    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.5,
        help="Jeda antarpermintaan ke Ollama dalam detik.",
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Membatasi jumlah data yang diklasifikasikan.",
    )

    parser.add_argument(
        "--n-per-tema",
        type=int,
        default=None,
        help="Jumlah data yang diambil untuk setiap tema.",
    )

    parser.add_argument(
        "--ambiguity-ratio",
        type=float,
        default=0.20,
        help=(
            "Proporsi data yang mendapat konteks tambahan ambigu. "
            "Nilai harus berada antara 0 dan 1."
        ),
    )

    parser.add_argument(
        "--easy-ratio",
        type=float,
        default=0.35,
        help="Proporsi data dengan tingkat kesulitan mudah.",
    )

    parser.add_argument(
        "--medium-ratio",
        type=float,
        default=0.40,
        help="Proporsi data dengan tingkat kesulitan sedang.",
    )

    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.25,
        help="Proporsi data dengan tingkat kesulitan sulit.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reproduksibilitas data.",
    )

    return parser.parse_args()


# =====================================================
# VALIDASI ARGUMEN
# =====================================================

def validate_args(args) -> None:
    if args.delay < 0:
        raise ValueError(
            "--delay tidak boleh bernilai negatif."
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit harus lebih besar dari 0."
        )

    if args.n_per_tema is not None and args.n_per_tema <= 0:
        raise ValueError(
            "--n-per-tema harus lebih besar dari 0."
        )

    if not 0 <= args.ambiguity_ratio <= 1:
        raise ValueError(
            "--ambiguity-ratio harus berada antara 0 dan 1."
        )

    difficulty_total = (
        args.easy_ratio
        + args.medium_ratio
        + args.hard_ratio
    )

    if abs(difficulty_total - 1.0) > 1e-9:
        raise ValueError(
            "Jumlah --easy-ratio, --medium-ratio, "
            "dan --hard-ratio harus sama dengan 1.0."
        )

    for argument_name, value in [
        ("easy-ratio", args.easy_ratio),
        ("medium-ratio", args.medium_ratio),
        ("hard-ratio", args.hard_ratio),
    ]:
        if not 0 <= value <= 1:
            raise ValueError(
                f"--{argument_name} harus berada antara 0 dan 1."
            )


# =====================================================
# BANNER
# =====================================================

def print_banner():
    print(
        """
================================================================
  KLASIFIKASI TEMA KESEHATAN BAYI DAN BALITA PADA TEKS LAPORAN
  MENGGUNAKAN LARGE LANGUAGE MODEL DAN ZERO-SHOT LEARNING
================================================================
"""
    )


# =====================================================
# INFORMASI DATASET
# =====================================================

def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RINGKASAN DATASET")
    print("=" * 60)

    print(f"Total data: {len(df)}")

    if "tema_aktual" in df.columns:
        print("\nDistribusi tema aktual:")

        for label, count in (
            df["tema_aktual"]
            .value_counts()
            .items()
        ):
            print(f"- {label}: {count}")

    if "tingkat_kesulitan" in df.columns:
        print("\nDistribusi tingkat kesulitan:")

        for level, count in (
            df["tingkat_kesulitan"]
            .value_counts()
            .items()
        ):
            print(f"- {level}: {count}")

    if "ambigu" in df.columns:
        total_ambigu = (
            df["ambigu"]
            .astype(bool)
            .sum()
        )

        percentage = (
            total_ambigu / len(df) * 100
            if len(df) > 0
            else 0
        )

        print(
            f"\nData ambigu: {total_ambigu}/{len(df)} "
            f"({percentage:.2f}%)"
        )


# =====================================================
# GENERATE DATA
# =====================================================

def step_generate(args) -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: GENERATE TEKS LAPORAN DARI DATA TABULAR")
    print("=" * 60)

    generator = TextGenerator(
        seed=args.seed,
        easy_ratio=args.easy_ratio,
        medium_ratio=args.medium_ratio,
        hard_ratio=args.hard_ratio,
    )

    df = generator.generate_and_save(
        n_per_tema=args.n_per_tema,
        shuffle=True,
        ambiguity_ratio=args.ambiguity_ratio,
    )

    print_dataset_summary(df)

    print("\nContoh teks per tema:")

    for tema in TEMA_LABELS:
        subset = df[
            df["tema_aktual"] == tema
        ]

        if subset.empty:
            continue

        sample = subset.iloc[0]

        print("\n" + "-" * 60)
        print(f"Tema aktual       : {tema}")

        if "tingkat_kesulitan" in sample:
            print(
                "Tingkat kesulitan : "
                f"{sample['tingkat_kesulitan']}"
            )

        if "ambigu" in sample:
            print(
                f"Ambigu             : {sample['ambigu']}"
            )

        print("Teks:")
        print(sample["teks_laporan"])

    return df


# =====================================================
# PEMBERSIHAN HASIL PREDIKSI
# =====================================================

def normalize_prediction_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    if "tema_aktual" in result.columns:
        result["tema_aktual"] = (
            result["tema_aktual"]
            .astype(str)
            .str.strip()
        )

    if "tema_prediksi" in result.columns:
        result["tema_prediksi"] = (
            result["tema_prediksi"]
            .astype(str)
            .str.strip()
        )

    return result


# =====================================================
# KLASIFIKASI DATA
# =====================================================

def step_classify(
    args,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2: KLASIFIKASI ZERO-SHOT DENGAN LLM")
    print("=" * 60)

    if df is None:
        df = load_generated_data()

    if df.empty:
        raise ValueError(
            "Data klasifikasi kosong."
        )

    if args.limit:
        df = df.head(args.limit).copy()

        print(
            f"[INFO] Data dibatasi menjadi "
            f"{len(df)} baris."
        )

    print(f"[INFO] Total data : {len(df)}")
    print(f"[INFO] Model      : {args.model}")
    print(f"[INFO] Reasoning  : {args.reasoning}")
    print(f"[INFO] Delay      : {args.delay} detik")

    classifier = OllamaClassifier(
        model=args.model
    )

    if not classifier.check_connection():
        print("\n[ERROR] Gagal terhubung ke Ollama.")
        print("Pastikan Ollama sudah berjalan:")
        print("  ollama serve")
        print(f"  ollama pull {args.model}")

        sys.exit(1)

    start_time = time.time()

    df_result = classifier.classify_dataframe(
        df,
        with_reasoning=args.reasoning,
        delay=args.delay,
    )

    total_time = (
        time.time() - start_time
    )

    if df_result.empty:
        raise ValueError(
            "Classifier tidak menghasilkan data."
        )

    if "tema_prediksi" not in df_result.columns:
        raise ValueError(
            "Kolom tema_prediksi tidak ditemukan "
            "pada hasil klasifikasi."
        )

    df_result = normalize_prediction_columns(
        df_result
    )

    print("\n" + "=" * 60)
    print("RINGKASAN HASIL KLASIFIKASI")
    print("=" * 60)

    print(f"Total data : {len(df_result)}")
    print(f"Waktu      : {total_time:.2f} detik")

    average_time = (
        total_time / len(df_result)
        if len(df_result) > 0
        else 0
    )

    print(
        f"Rata-rata  : "
        f"{average_time:.2f} detik/teks"
    )

    print("\nDistribusi prediksi:")

    for tema, count in (
        df_result["tema_prediksi"]
        .value_counts(dropna=False)
        .items()
    ):
        print(f"- {tema}: {count}")

    if "tema_aktual" in df_result.columns:
        correct_mask = (
            df_result["tema_prediksi"]
            == df_result["tema_aktual"]
        )

        correct = int(
            correct_mask.sum()
        )

        incorrect = (
            len(df_result) - correct
        )

        accuracy = (
            correct / len(df_result) * 100
            if len(df_result) > 0
            else 0
        )

        print(
            f"\nPrediksi benar : "
            f"{correct}/{len(df_result)}"
        )

        print(
            f"Prediksi salah : "
            f"{incorrect}/{len(df_result)}"
        )

        print(
            f"Akurasi        : "
            f"{accuracy:.2f}%"
        )

        if incorrect > 0:
            print("\nContoh prediksi salah:")

            error_columns = [
                "teks_laporan",
                "tema_aktual",
                "tema_prediksi",
            ]

            optional_columns = [
                "tingkat_kesulitan",
                "ambigu",
            ]

            for column in optional_columns:
                if column in df_result.columns:
                    error_columns.append(column)

            print(
                df_result.loc[
                    ~correct_mask,
                    error_columns,
                ]
                .head(10)
                .to_string(index=False)
            )

        else:
            print(
                "\n[WARNING] Seluruh prediksi benar. "
                "Dataset mungkin masih terlalu mudah."
            )

    output_directory = os.path.dirname(
        args.output
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    df_result.to_csv(
        args.output,
        index=False,
    )

    print(
        f"\n[INFO] Hasil klasifikasi "
        f"disimpan ke: {args.output}"
    )

    return df_result


# =====================================================
# KLASIFIKASI SATU TEKS
# =====================================================

def classify_single_text(args) -> None:
    classifier = OllamaClassifier(
        model=args.model
    )

    if not classifier.check_connection():
        print("[ERROR] Ollama belum aktif.")
        print("Jalankan: ollama serve")

        sys.exit(1)

    result = classifier.classify_single(
        args.single,
        with_reasoning=args.reasoning,
    )

    print("\n" + "=" * 60)
    print("HASIL KLASIFIKASI")
    print("=" * 60)

    print(f"Teks  : {args.single}")
    print(
        f"Tema  : "
        f"{result.get('tema_prediksi', '-')}"
    )
    print(
        f"Waktu : "
        f"{result.get('waktu_proses', '-')} detik"
    )

    if args.reasoning:
        print(
            f"Alasan: "
            f"{result.get('alasan', '-')}"
        )


# =====================================================
# MAIN
# =====================================================

def main():
    print_banner()

    args = parse_args()

    try:
        validate_args(args)

        if args.single:
            classify_single_text(args)
            return

        if args.generate:
            step_generate(args)

            print(
                "\n[INFO] Generate selesai."
            )
            print(
                f"[INFO] File: {GENERATED_DATA_PATH}"
            )

            return

        if args.classify:
            step_classify(args)

            print(
                "\n[INFO] Klasifikasi selesai."
            )
            print(
                "[INFO] Jalankan: python evaluate.py"
            )

            return

        print(
            "[INFO] PIPELINE LENGKAP: "
            "Generate -> Klasifikasi\n"
        )

        generated_df = step_generate(args)

        step_classify(
            args,
            generated_df,
        )

        print("\n[INFO] Pipeline selesai.")
        print(
            "[INFO] Jalankan: "
            "python evaluate.py"
        )

    except (
        ValueError,
        FileNotFoundError,
        KeyError,
    ) as error:
        print(
            f"\n[ERROR] {error}"
        )

        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\n[INFO] Program dihentikan pengguna."
        )

        sys.exit(130)


if __name__ == "__main__":
    main()

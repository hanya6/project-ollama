import os
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    OUTPUT_CSV,
    RESULTS_DIR,
    TEMA_LABELS,
)


# =====================================================
# GITHUB CONFIG
# =====================================================

GITHUB_USERNAME = "hanya6"
GITHUB_REPO = "project-ollama"
GITHUB_BRANCH = "feature/klasifikasi-tema-posyandu"

GIT_EMAIL = "sedekah100m1@gmail.com"
GIT_NAME = "ya"


# =====================================================
# GIT COMMAND
# =====================================================

def run_cmd(command: str) -> bool:
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        return True

    except subprocess.CalledProcessError as error:
        print("\n[ERROR] Perintah gagal:")
        print(command)

        if error.stdout:
            print(error.stdout)

        if error.stderr:
            print(error.stderr)

        return False


# =====================================================
# NORMALISASI LABEL
# =====================================================

def normalize_label_series(
    series: pd.Series,
) -> pd.Series:
    """
    Membersihkan spasi pada label tanpa mengubah nama label.
    """
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


# =====================================================
# VALIDASI DATA EVALUASI
# =====================================================

def validate_evaluation_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "teks_laporan",
        "tema_aktual",
        "tema_prediksi",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            "Data hasil klasifikasi kosong."
        )

    result = df.copy()

    result["teks_laporan"] = (
        result["teks_laporan"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["tema_aktual"] = normalize_label_series(
        result["tema_aktual"]
    )

    result["tema_prediksi"] = normalize_label_series(
        result["tema_prediksi"]
    )

    empty_text = (
        result["teks_laporan"] == ""
    ).sum()

    empty_actual = (
        result["tema_aktual"] == ""
    ).sum()

    empty_prediction = (
        result["tema_prediksi"] == ""
    ).sum()

    if empty_text > 0:
        raise ValueError(
            f"Ditemukan {empty_text} teks laporan kosong."
        )

    if empty_actual > 0:
        raise ValueError(
            f"Ditemukan {empty_actual} tema aktual kosong."
        )

    if empty_prediction > 0:
        raise ValueError(
            f"Ditemukan {empty_prediction} tema prediksi kosong."
        )

    unknown_actual = sorted(
        set(result["tema_aktual"])
        - set(TEMA_LABELS)
    )

    unknown_prediction = sorted(
        set(result["tema_prediksi"])
        - set(TEMA_LABELS)
    )

    if unknown_actual:
        print(
            "[WARNING] Label aktual di luar TEMA_LABELS:"
        )

        for label in unknown_actual:
            print(f"- {label}")

    if unknown_prediction:
        print(
            "[WARNING] Label prediksi di luar TEMA_LABELS:"
        )

        for label in unknown_prediction:
            print(f"- {label}")

    return result


# =====================================================
# RINGKASAN VALIDASI
# =====================================================

def print_validation_summary(
    df: pd.DataFrame,
) -> pd.Series:
    correct_mask = (
        df["tema_aktual"]
        == df["tema_prediksi"]
    )

    total = len(df)
    correct = int(correct_mask.sum())
    incorrect = total - correct

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("PEMERIKSAAN DATA EVALUASI")
    print("=" * 60)

    print(f"Total data       : {total}")
    print(f"Prediksi benar   : {correct}")
    print(f"Prediksi salah   : {incorrect}")
    print(f"Akurasi awal     : {accuracy:.2f}%")

    print("\nDistribusi tema aktual:")

    print(
        df["tema_aktual"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDistribusi tema prediksi:")

    print(
        df["tema_prediksi"]
        .value_counts(dropna=False)
        .to_string()
    )

    if incorrect == 0:
        print(
            "\n[WARNING] Seluruh prediksi sama dengan label aktual."
        )
        print(
            "[WARNING] Periksa apakah dataset terlalu mudah, "
            "template membocorkan nama tema, atau prompt "
            "terlalu eksplisit."
        )
    else:
        print("\nContoh prediksi salah:")

        error_columns = [
            "teks_laporan",
            "tema_aktual",
            "tema_prediksi",
        ]

        for optional_column in [
            "tingkat_kesulitan",
            "ambigu",
            "alasan",
        ]:
            if optional_column in df.columns:
                error_columns.append(
                    optional_column
                )

        print(
            df.loc[
                ~correct_mask,
                error_columns,
            ]
            .head(20)
            .to_string(index=False)
        )

    return correct_mask


# =====================================================
# PUSH RESULTS TO GITHUB
# =====================================================

def push_results_to_github():
    print("\n" + "=" * 60)
    print("UPDATE HASIL EVALUASI KE GITHUB")
    print("=" * 60)

    token = os.environ.get(
        "GITHUB_TOKEN"
    )

    if not token:
        print(
            "[INFO] GITHUB_TOKEN tidak tersedia."
        )
        print(
            "[INFO] Hasil evaluasi tidak dipush ke GitHub."
        )
        return

    if not os.path.exists(".git"):
        print(
            "[WARNING] Folder aktif bukan repository Git."
        )
        return

    remote_url = (
        f"https://{GITHUB_USERNAME}:{token}"
        f"@github.com/{GITHUB_USERNAME}/"
        f"{GITHUB_REPO}.git"
    )

    commands = [
        (
            "Konfigurasi email Git",
            f'git config --global user.email "{GIT_EMAIL}"',
        ),
        (
            "Konfigurasi nama Git",
            f'git config --global user.name "{GIT_NAME}"',
        ),
        (
            "Mengatur remote Git",
            f"git remote set-url origin {remote_url}",
        ),
        (
            "Checkout branch",
            f"git checkout {GITHUB_BRANCH}",
        ),
    ]

    for description, command in commands:
        print(f"[INFO] {description}")

        if not run_cmd(command):
            print(
                "[ERROR] Proses Git dihentikan."
            )
            return

    run_cmd("git add evaluate.py")
    run_cmd("git add -f results/*.csv")
    run_cmd("git add -f results/*.png")
    run_cmd("git add -f results/*.txt")

    status = subprocess.run(
        "git status --porcelain",
        shell=True,
        text=True,
        capture_output=True,
    )

    if status.stdout.strip() == "":
        print(
            "[INFO] Tidak ada perubahan baru untuk commit."
        )
        return

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    commit_message = (
        f"update evaluation results {timestamp}"
    )

    commit_ok = run_cmd(
        f'git commit -m "{commit_message}"'
    )

    if not commit_ok:
        print("[ERROR] Commit gagal.")
        return

    pull_ok = run_cmd(
        f"git pull origin "
        f"{GITHUB_BRANCH} --rebase"
    )

    if not pull_ok:
        print(
            "[ERROR] Pull rebase gagal."
        )
        return

    push_ok = run_cmd(
        f"git push origin "
        f"{GITHUB_BRANCH}"
    )

    if push_ok:
        print(
            "\n[INFO] Hasil evaluasi berhasil "
            "dipush ke GitHub."
        )
    else:
        print(
            "\n[ERROR] Push ke GitHub gagal."
        )


# =====================================================
# MAIN EVALUATION
# =====================================================

def main():
    print("=" * 60)
    print("EVALUASI HASIL KLASIFIKASI")
    print("=" * 60)

    if not os.path.exists(OUTPUT_CSV):
        raise FileNotFoundError(
            "File hasil klasifikasi tidak ditemukan: "
            f"{OUTPUT_CSV}. Jalankan terlebih dahulu: "
            "python main.py --classify"
        )

    print(
        f"[INFO] Membaca hasil klasifikasi: "
        f"{OUTPUT_CSV}"
    )

    df = pd.read_csv(
        OUTPUT_CSV
    )

    df = validate_evaluation_data(
        df
    )

    correct_mask = print_validation_summary(
        df
    )

    y_true = df["tema_aktual"]
    y_pred = df["tema_prediksi"]

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    # =====================================================
    # AKURASI
    # =====================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 60)
    print("HASIL EVALUASI")
    print("=" * 60)

    print(
        f"\nAkurasi: "
        f"{accuracy * 100:.2f}%"
    )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=TEMA_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    print("\nClassification Report:")
    print(report_df)

    report_csv_path = os.path.join(
        RESULTS_DIR,
        "classification_report.csv",
    )

    report_df.to_csv(
        report_csv_path
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=TEMA_LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Aktual {label}"
            for label in TEMA_LABELS
        ],
        columns=[
            f"Prediksi {label}"
            for label in TEMA_LABELS
        ],
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    cm_csv_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.csv",
    )

    cm_df.to_csv(
        cm_csv_path
    )

    # =====================================================
    # GAMBAR CONFUSION MATRIX
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TEMA_LABELS,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=30,
    )

    plt.title(
        "Confusion Matrix Klasifikasi Tema Posyandu"
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()

    cm_img_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png",
    )

    plt.savefig(
        cm_img_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =====================================================
    # CLASSIFICATION REPORT PER KELAS
    # =====================================================

    per_class_df = report_df.loc[
        TEMA_LABELS,
        [
            "precision",
            "recall",
            "f1-score",
        ],
    ]

    per_class_df.plot(
        kind="bar",
        figsize=(11, 7),
    )

    plt.title(
        "Perbandingan Metrik Evaluasi per Kelas"
    )
    plt.xlabel("Kelas")
    plt.ylabel("Nilai")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25)
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.7,
    )
    plt.legend(title="Metrik")
    plt.tight_layout()

    report_img_path = os.path.join(
        RESULTS_DIR,
        "classification_report_per_kelas.png",
    )

    plt.savefig(
        report_img_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =====================================================
    # TABEL HASIL KLASIFIKASI
    # =====================================================

    result_columns = [
        "teks_laporan",
        "tema_aktual",
        "tema_prediksi",
    ]

    for optional_column in [
        "tingkat_kesulitan",
        "ambigu",
        "alasan",
        "waktu_proses",
    ]:
        if optional_column in df.columns:
            result_columns.append(
                optional_column
            )

    hasil_df = df[
        result_columns
    ].copy()

    hasil_df["hasil"] = (
        correct_mask.map({
            True: "Benar",
            False: "Salah",
        })
    )

    print("\nTabel hasil klasifikasi:")
    print(
        hasil_df
        .head(20)
        .to_string(index=False)
    )

    hasil_csv_path = os.path.join(
        RESULTS_DIR,
        "tabel_hasil_klasifikasi.csv",
    )

    hasil_df.to_csv(
        hasil_csv_path,
        index=False,
    )

    # =====================================================
    # DATA SALAH KLASIFIKASI
    # =====================================================

    error_df = hasil_df[
        hasil_df["hasil"] == "Salah"
    ].copy()

    error_csv_path = os.path.join(
        RESULTS_DIR,
        "kesalahan_klasifikasi.csv",
    )

    error_df.to_csv(
        error_csv_path,
        index=False,
    )

    print(
        f"\nTotal kesalahan klasifikasi: "
        f"{len(error_df)}"
    )

    # =====================================================
    # DISTRIBUSI TEMA AKTUAL
    # =====================================================

    actual_distribution = (
        df["tema_aktual"]
        .value_counts()
        .reindex(
            TEMA_LABELS,
            fill_value=0,
        )
    )

    print("\nDistribusi Tema Aktual:")
    print(actual_distribution)

    actual_distribution.plot(
        kind="bar",
        figsize=(9, 6),
    )

    plt.title("Distribusi Tema Aktual")
    plt.xlabel("Tema")
    plt.ylabel("Jumlah Data")
    plt.xticks(rotation=25)
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.7,
    )
    plt.tight_layout()

    actual_distribution_path = os.path.join(
        RESULTS_DIR,
        "distribusi_tema_aktual.png",
    )

    plt.savefig(
        actual_distribution_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =====================================================
    # DISTRIBUSI TEMA PREDIKSI
    # =====================================================

    prediction_distribution = (
        df["tema_prediksi"]
        .value_counts()
        .reindex(
            TEMA_LABELS,
            fill_value=0,
        )
    )

    print("\nDistribusi Tema Prediksi:")
    print(prediction_distribution)

    prediction_distribution.plot(
        kind="bar",
        figsize=(9, 6),
    )

    plt.title("Distribusi Tema Prediksi")
    plt.xlabel("Tema")
    plt.ylabel("Jumlah Data")
    plt.xticks(rotation=25)
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.7,
    )
    plt.tight_layout()

    prediction_distribution_path = os.path.join(
        RESULTS_DIR,
        "distribusi_tema_prediksi.png",
    )

    plt.savefig(
        prediction_distribution_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =====================================================
    # PERBANDINGAN DISTRIBUSI
    # =====================================================

    distribution_df = pd.DataFrame({
        "Aktual": actual_distribution,
        "Prediksi": prediction_distribution,
    })

    distribution_csv_path = os.path.join(
        RESULTS_DIR,
        "distribusi_tema.csv",
    )

    distribution_df.to_csv(
        distribution_csv_path
    )

    distribution_df.plot(
        kind="bar",
        figsize=(11, 7),
    )

    plt.title(
        "Perbandingan Distribusi Tema Aktual dan Prediksi"
    )
    plt.xlabel("Tema")
    plt.ylabel("Jumlah Data")
    plt.xticks(rotation=25)
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.7,
    )
    plt.legend(title="Jenis")
    plt.tight_layout()

    comparison_path = os.path.join(
        RESULTS_DIR,
        "distribusi_tema_aktual_vs_prediksi.png",
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =====================================================
    # EVALUASI BERDASARKAN TINGKAT KESULITAN
    # =====================================================

    difficulty_evaluation_path = None

    if "tingkat_kesulitan" in df.columns:
        difficulty_rows = []

        for difficulty, subset in df.groupby(
            "tingkat_kesulitan"
        ):
            difficulty_accuracy = accuracy_score(
                subset["tema_aktual"],
                subset["tema_prediksi"],
            )

            difficulty_rows.append({
                "tingkat_kesulitan": difficulty,
                "jumlah_data": len(subset),
                "jumlah_benar": int(
                    (
                        subset["tema_aktual"]
                        == subset["tema_prediksi"]
                    ).sum()
                ),
                "jumlah_salah": int(
                    (
                        subset["tema_aktual"]
                        != subset["tema_prediksi"]
                    ).sum()
                ),
                "akurasi": difficulty_accuracy,
                "akurasi_persen": (
                    difficulty_accuracy * 100
                ),
            })

        difficulty_df = pd.DataFrame(
            difficulty_rows
        )

        print(
            "\nEvaluasi berdasarkan tingkat kesulitan:"
        )
        print(
            difficulty_df.to_string(
                index=False
            )
        )

        difficulty_evaluation_path = os.path.join(
            RESULTS_DIR,
            "evaluasi_tingkat_kesulitan.csv",
        )

        difficulty_df.to_csv(
            difficulty_evaluation_path,
            index=False,
        )

    # =====================================================
    # EVALUASI DATA AMBIGU
    # =====================================================

    ambiguity_evaluation_path = None

    if "ambigu" in df.columns:
        ambiguity_df = df.copy()

        ambiguity_df["ambigu"] = (
            ambiguity_df["ambigu"]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
            .fillna(False)
        )

        ambiguity_rows = []

        for ambiguous, subset in ambiguity_df.groupby(
            "ambigu"
        ):
            ambiguity_accuracy = accuracy_score(
                subset["tema_aktual"],
                subset["tema_prediksi"],
            )

            ambiguity_rows.append({
                "jenis_data": (
                    "Ambigu"
                    if ambiguous
                    else "Tidak Ambigu"
                ),
                "jumlah_data": len(subset),
                "jumlah_benar": int(
                    (
                        subset["tema_aktual"]
                        == subset["tema_prediksi"]
                    ).sum()
                ),
                "jumlah_salah": int(
                    (
                        subset["tema_aktual"]
                        != subset["tema_prediksi"]
                    ).sum()
                ),
                "akurasi": ambiguity_accuracy,
                "akurasi_persen": (
                    ambiguity_accuracy * 100
                ),
            })

        ambiguity_result_df = pd.DataFrame(
            ambiguity_rows
        )

        print(
            "\nEvaluasi berdasarkan status ambigu:"
        )
        print(
            ambiguity_result_df.to_string(
                index=False
            )
        )

        ambiguity_evaluation_path = os.path.join(
            RESULTS_DIR,
            "evaluasi_data_ambigu.csv",
        )

        ambiguity_result_df.to_csv(
            ambiguity_evaluation_path,
            index=False,
        )

    # =====================================================
    # SIMPAN RINGKASAN TXT
    # =====================================================

    summary_path = os.path.join(
        RESULTS_DIR,
        "evaluasi_ringkasan.txt",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "EVALUASI HASIL KLASIFIKASI\n"
        )
        file.write("=" * 70 + "\n")

        file.write(
            f"Total data       : {len(df)}\n"
        )
        file.write(
            f"Prediksi benar   : "
            f"{int(correct_mask.sum())}\n"
        )
        file.write(
            f"Prediksi salah   : "
            f"{int((~correct_mask).sum())}\n"
        )
        file.write(
            f"Akurasi          : "
            f"{accuracy * 100:.2f}%\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )
        file.write("-" * 70 + "\n")
        file.write(
            report_df.to_string()
        )

        file.write(
            "\n\nCONFUSION MATRIX\n"
        )
        file.write("-" * 70 + "\n")
        file.write(
            cm_df.to_string()
        )

        file.write(
            "\n\nDISTRIBUSI TEMA\n"
        )
        file.write("-" * 70 + "\n")
        file.write(
            distribution_df.to_string()
        )

        file.write(
            "\n\nCONTOH KESALAHAN KLASIFIKASI\n"
        )
        file.write("-" * 70 + "\n")

        if error_df.empty:
            file.write(
                "Tidak ditemukan kesalahan klasifikasi."
            )
        else:
            file.write(
                error_df.head(20).to_string(
                    index=False
                )
            )

    # =====================================================
    # INFO FILE
    # =====================================================

    print("\n" + "=" * 60)
    print("FILE HASIL EVALUASI")
    print("=" * 60)

    output_files = [
        report_csv_path,
        cm_csv_path,
        cm_img_path,
        report_img_path,
        hasil_csv_path,
        error_csv_path,
        distribution_csv_path,
        actual_distribution_path,
        prediction_distribution_path,
        comparison_path,
        summary_path,
    ]

    if difficulty_evaluation_path:
        output_files.append(
            difficulty_evaluation_path
        )

    if ambiguity_evaluation_path:
        output_files.append(
            ambiguity_evaluation_path
        )

    for output_file in output_files:
        print(f"- {output_file}")

    # =====================================================
    # PUSH KE GITHUB
    # =====================================================

    push_results_to_github()


if __name__ == "__main__":
    try:
        main()

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
    ) as error:
        print(
            f"\n[ERROR] {error}"
        )

        raise SystemExit(1)

    except KeyboardInterrupt:
        print(
            "\n[INFO] Evaluasi dihentikan pengguna."
        )

        raise SystemExit(130)

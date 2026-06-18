import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import OUTPUT_CSV, RESULTS_DIR, TEMA_LABELS


GITHUB_BRANCH = "feature/klasifikasi-tema-posyandu"


def run_git_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        return True

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Perintah gagal: {command}")

        if e.stdout:
            print(e.stdout)

        if e.stderr:
            print(e.stderr)

        return False


def push_results_to_github():
    print("\n" + "=" * 60)
    print("UPDATE HASIL EVALUASI KE GITHUB")
    print("=" * 60)

    if not os.path.exists(".git"):
        print("[WARNING] Folder ini bukan repository Git.")
        return

    run_git_command("git status")

    run_git_command("git add results/")

    commit_success = run_git_command(
        'git commit -m "update evaluation results"'
    )

    if not commit_success:
        print("[INFO] Tidak ada perubahan baru untuk di-commit.")
        return

    push_success = run_git_command(
        f"git push origin {GITHUB_BRANCH}"
    )

    if push_success:
        print(f"[INFO] Hasil evaluasi berhasil di-push ke branch {GITHUB_BRANCH}.")
    else:
        print("[WARNING] Push gagal. Cek token GitHub, remote, atau branch.")


def main():
    print("=" * 60)
    print("EVALUASI HASIL KLASIFIKASI")
    print("=" * 60)

    if not os.path.exists(OUTPUT_CSV):
        raise FileNotFoundError(
            f"File hasil klasifikasi tidak ditemukan: {OUTPUT_CSV}. "
            "Jalankan terlebih dahulu: python main.py --classify"
        )

    df = pd.read_csv(OUTPUT_CSV)

    if "tema_aktual" not in df.columns or "tema_prediksi" not in df.columns:
        raise ValueError(
            "File harus memiliki kolom tema_aktual dan tema_prediksi."
        )

    y_true = df["tema_aktual"]
    y_pred = df["tema_prediksi"]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nAkurasi: {accuracy * 100:.2f}%")

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=TEMA_LABELS,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    print("\nClassification Report:")
    print(report_df)

    report_csv_path = os.path.join(
        RESULTS_DIR,
        "classification_report.csv"
    )

    report_df.to_csv(report_csv_path)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=TEMA_LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual {label}" for label in TEMA_LABELS],
        columns=[f"Prediksi {label}" for label in TEMA_LABELS]
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    cm_csv_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.csv"
    )

    cm_df.to_csv(cm_csv_path)

    plt.figure(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TEMA_LABELS
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
        xticks_rotation=30
    )

    plt.title("Confusion Matrix Klasifikasi Tema Laporan Posyandu")
    plt.tight_layout()

    cm_img_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(cm_img_path, dpi=300)
    plt.show()
    plt.close()

    per_class_df = report_df.loc[
        TEMA_LABELS,
        ["precision", "recall", "f1-score"]
    ]

    ax = per_class_df.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Perbandingan Classification Report per Kelas")
    plt.xlabel("Kelas")
    plt.ylabel("Nilai")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30)
    plt.legend(title="Metrik")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    report_img_path = os.path.join(
        RESULTS_DIR,
        "classification_report_per_kelas.png"
    )

    plt.savefig(report_img_path, dpi=300)
    plt.show()
    plt.close()

    summary_path = os.path.join(
        RESULTS_DIR,
        "evaluasi_ringkasan.txt"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("EVALUASI HASIL KLASIFIKASI\n")
        f.write("=" * 60 + "\n")
        f.write(f"Akurasi: {accuracy * 100:.2f}%\n\n")

        f.write("Classification Report:\n")
        f.write(report_df.to_string())

        f.write("\n\nConfusion Matrix:\n")
        f.write(cm_df.to_string())

    print("\n[INFO] File evaluasi berhasil disimpan:")
    print(f"- {cm_csv_path}")
    print(f"- {report_csv_path}")
    print(f"- {cm_img_path}")
    print(f"- {report_img_path}")
    print(f"- {summary_path}")

    push_results_to_github()


if __name__ == "__main__":
    main()

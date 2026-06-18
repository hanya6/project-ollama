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


GITHUB_USERNAME = "hanya6"
GITHUB_REPO = "project-ollama"
GITHUB_BRANCH = "feature/klasifikasi-tema-posyandu"

GIT_EMAIL = "sedekah100m1@gmail.com"
GIT_NAME = "ya"


def run_cmd(command):
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
        print(f"\n[ERROR] Perintah gagal:")
        print(command)

        if e.stdout:
            print(e.stdout)

        if e.stderr:
            print(e.stderr)

        return False


def push_results_to_github():
    print("\n" + "=" * 60)
    print("UPDATE HASIL EVALUASI KE GITHUB")
    print("=" * 60)

    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        print("[ERROR] GITHUB_TOKEN belum tersedia.")
        print('Jalankan dulu: export GITHUB_TOKEN="token_github_anda"')
        return

    if not os.path.exists(".git"):
        print("[ERROR] Folder ini bukan repository Git.")
        return

    remote_url = (
        f"https://{GITHUB_USERNAME}:{token}"
        f"@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
    )

    run_cmd(f'git config --global user.email "{GIT_EMAIL}"')
    run_cmd(f'git config --global user.name "{GIT_NAME}"')

    run_cmd(f"git remote set-url origin {remote_url}")

    run_cmd(f"git checkout {GITHUB_BRANCH}")

    # Simpan perubahan evaluate.py jika ada
    run_cmd("git add evaluate.py")

    # Paksa add results karena results/*.csv, *.png, *.txt ada di .gitignore
    run_cmd("git add -f results/*.csv")
    run_cmd("git add -f results/*.png")
    run_cmd("git add -f results/*.txt")

    status = subprocess.run(
        "git status --porcelain",
        shell=True,
        text=True,
        capture_output=True
    )

    if status.stdout.strip() == "":
        print("[INFO] Tidak ada perubahan baru untuk commit.")
        run_cmd(f"git push origin {GITHUB_BRANCH}")
        return

    run_cmd('git commit -m "update evaluation results"')

    # Ambil update remote dulu agar tidak rejected fetch first
    pull_ok = run_cmd(
        f"git pull origin {GITHUB_BRANCH} --rebase"
    )

    if not pull_ok:
        print("[ERROR] Pull rebase gagal. Selesaikan konflik terlebih dahulu.")
        return

    push_ok = run_cmd(
        f"git push origin {GITHUB_BRANCH}"
    )

    if push_ok:
        print("\n[INFO] BERHASIL PUSH RESULTS KE GITHUB")
        print(f"Repository: {GITHUB_USERNAME}/{GITHUB_REPO}")
        print(f"Branch    : {GITHUB_BRANCH}")
        print("Folder    : results/")
    else:
        print("\n[ERROR] Push gagal.")


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

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TEMA_LABELS
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=30
    )

    plt.title("Confusion Matrix Klasifikasi Tema Posyandu")
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
    plt.xticks(rotation=20)
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
    print(f"- {report_csv_path}")
    print(f"- {cm_csv_path}")
    print(f"- {cm_img_path}")
    print(f"- {report_img_path}")
    print(f"- {summary_path}")

    push_results_to_github()


if __name__ == "__main__":
    main()

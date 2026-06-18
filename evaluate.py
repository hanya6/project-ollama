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

# =====================================================
# GITHUB CONFIG
# =====================================================

GITHUB_BRANCH = "feature/klasifikasi-tema-posyandu"

GIT_EMAIL = "sedekah100m1@gmail.com"
GIT_USERNAME = "hanya6"


# =====================================================
# GIT HELPER
# =====================================================

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

        print(f"\n[ERROR] {command}")

        if e.stdout:
            print(e.stdout)

        if e.stderr:
            print(e.stderr)

        return False


# =====================================================
# PUSH TO GITHUB
# =====================================================

def push_results_to_github():

    print("\n" + "=" * 60)
    print("UPDATE HASIL EVALUASI KE GITHUB")
    print("=" * 60)

    if not os.path.exists(".git"):
        print("[WARNING] Folder ini bukan repository git.")
        return

    run_git_command(
        f'git config --global user.email "{GIT_EMAIL}"'
    )

    run_git_command(
        f'git config --global user.name "{GIT_USERNAME}"'
    )

    run_git_command(
        f"git checkout {GITHUB_BRANCH}"
    )

    # force add karena di-ignore .gitignore
    run_git_command("git add -f results/*.csv")
    run_git_command("git add -f results/*.png")
    run_git_command("git add -f results/*.txt")

    status = subprocess.run(
        "git status --porcelain",
        shell=True,
        text=True,
        capture_output=True
    )

    if status.stdout.strip() == "":
        print(
            "[INFO] Tidak ada perubahan baru."
        )
        print(
            "[INFO] GitHub sudah memiliki file terbaru."
        )
        return

    commit_success = run_git_command(
        'git commit -m "update evaluation results"'
    )

    if not commit_success:
        return

    push_success = run_git_command(
        f"git push origin {GITHUB_BRANCH}"
    )

    if push_success:

        print("\n[INFO] BERHASIL PUSH KE GITHUB")
        print(f"Branch : {GITHUB_BRANCH}")
        print("Folder : results/")

    else:

        print("\n[ERROR] Push gagal.")


# =====================================================
# EVALUATION
# =====================================================

def main():

    print("=" * 60)
    print("EVALUASI HASIL KLASIFIKASI")
    print("=" * 60)

    if not os.path.exists(OUTPUT_CSV):
        raise FileNotFoundError(
            f"File hasil klasifikasi tidak ditemukan: {OUTPUT_CSV}"
        )

    df = pd.read_csv(OUTPUT_CSV)

    y_true = df["tema_aktual"]
    y_pred = df["tema_prediksi"]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =====================================================
    # ACCURACY
    # =====================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"\nAkurasi: {accuracy*100:.2f}%"
    )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=TEMA_LABELS,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    print("\nClassification Report:")
    print(report_df)

    report_csv = os.path.join(
        RESULTS_DIR,
        "classification_report.csv"
    )

    report_df.to_csv(report_csv)

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=TEMA_LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual {x}" for x in TEMA_LABELS],
        columns=[f"Prediksi {x}" for x in TEMA_LABELS]
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    cm_csv = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.csv"
    )

    cm_df.to_csv(cm_csv)

    # =====================================================
    # CONFUSION MATRIX IMAGE
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TEMA_LABELS
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    cm_img = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        cm_img,
        dpi=300
    )

    plt.show()

    # =====================================================
    # CLASSIFICATION REPORT CHART
    # =====================================================

    per_class_df = report_df.loc[
        TEMA_LABELS,
        ["precision", "recall", "f1-score"]
    ]

    ax = per_class_df.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(
        "Perbandingan Classification Report per Kelas"
    )

    plt.xlabel("Kelas")
    plt.ylabel("Nilai")

    plt.ylim(0, 1.05)

    plt.xticks(rotation=20)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.7
    )

    plt.tight_layout()

    report_img = os.path.join(
        RESULTS_DIR,
        "classification_report_per_kelas.png"
    )

    plt.savefig(
        report_img,
        dpi=300
    )

    plt.show()

    # =====================================================
    # SUMMARY TXT
    # =====================================================

    summary_file = os.path.join(
        RESULTS_DIR,
        "evaluasi_ringkasan.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "EVALUASI HASIL KLASIFIKASI\n"
        )

        f.write(
            "="*60 + "\n"
        )

        f.write(
            f"Akurasi: {accuracy*100:.2f}%\n\n"
        )

        f.write(
            report_df.to_string()
        )

        f.write(
            "\n\n"
        )

        f.write(
            cm_df.to_string()
        )

    print("\n[INFO] File evaluasi berhasil disimpan.")

    print(report_csv)
    print(cm_csv)
    print(cm_img)
    print(report_img)
    print(summary_file)

    # =====================================================
    # PUSH GITHUB
    # =====================================================

    push_results_to_github()


if __name__ == "__main__":
    main()

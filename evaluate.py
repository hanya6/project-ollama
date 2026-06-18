import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import OUTPUT_CSV, RESULTS_DIR, TEMA_LABELS


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

    y_true = df["tema_aktual"]
    y_pred = df["tema_prediksi"]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ============================
    # AKURASI
    # ============================
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nAkurasi: {accuracy * 100:.2f}%")

    # ============================
    # CLASSIFICATION REPORT
    # ============================
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

    report_csv_path = os.path.join(RESULTS_DIR, "classification_report.csv")
    report_df.to_csv(report_csv_path)

    # ============================
    # CONFUSION MATRIX
    # ============================
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

    cm_csv_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)

    # ============================
    # GAMBAR CONFUSION MATRIX
    # ============================
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

    cm_img_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=300)
    plt.show()

    # ============================
    # GRAFIK PERBANDINGAN REPORT PER KELAS
    # ============================
    per_class_df = report_df.loc[TEMA_LABELS, ["precision", "recall", "f1-score"]]

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

    report_img_path = os.path.join(RESULTS_DIR, "classification_report_per_kelas.png")
    plt.savefig(report_img_path, dpi=300)
    plt.show()

    # ============================
    # SIMPAN RINGKASAN TXT
    # ============================
    summary_path = os.path.join(RESULTS_DIR, "evaluasi_ringkasan.txt")

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


if __name__ == "__main__":
    main()

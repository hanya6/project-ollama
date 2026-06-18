import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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

    if "tema_aktual" not in df.columns or "tema_prediksi" not in df.columns:
        raise ValueError(
            "File hasil klasifikasi harus memiliki kolom tema_aktual dan tema_prediksi."
        )

    y_true = df["tema_aktual"]
    y_pred = df["tema_prediksi"]

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nAkurasi: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=TEMA_LABELS,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=TEMA_LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual_{label}" for label in TEMA_LABELS],
        columns=[f"Prediksi_{label}" for label in TEMA_LABELS]
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    eval_path = os.path.join(RESULTS_DIR, "evaluasi_confusion_matrix.csv")
    cm_df.to_csv(eval_path)

    report_path = os.path.join(RESULTS_DIR, "evaluasi_ringkasan.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("EVALUASI HASIL KLASIFIKASI\n")
        f.write("=" * 60 + "\n")
        f.write(f"Akurasi: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(
            classification_report(
                y_true,
                y_pred,
                labels=TEMA_LABELS,
                zero_division=0
            )
        )
        f.write("\n\nConfusion Matrix:\n")
        f.write(cm_df.to_string())

    print(f"\n[INFO] Confusion matrix disimpan: {eval_path}")
    print(f"[INFO] Ringkasan evaluasi disimpan: {report_path}")


if __name__ == "__main__":
    main()

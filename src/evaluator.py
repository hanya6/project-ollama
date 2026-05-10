"""
Modul Evaluasi Hasil Klasifikasi
Menghitung metrik dan menghasilkan visualisasi.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from tabulate import tabulate

from config import (
    TEMA_LABELS, RESULTS_DIR,
    OUTPUT_CONFUSION_MATRIX, OUTPUT_CLASSIFICATION_REPORT, OUTPUT_REPORT,
)


class ClassificationEvaluator:
    """Evaluator performa klasifikasi tema laporan posyandu."""

    def __init__(self, labels: list = None):
        self.labels = labels or TEMA_LABELS
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

    def calculate_metrics(self, y_true: list, y_pred: list) -> dict:
        """Hitung semua metrik evaluasi."""
        valid = [l for l in self.labels if l in set(y_true) | set(y_pred)]
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, labels=valid, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, labels=valid, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, labels=valid, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, labels=valid, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, labels=valid, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, labels=valid, average="macro", zero_division=0),
            "total_samples": len(y_true),
            "correct": sum(1 for t, p in zip(y_true, y_pred) if t == p),
        }

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Buat confusion matrix."""
        valid = [l for l in self.labels if l in set(y_true) | set(y_pred)]
        cm = confusion_matrix(y_true, y_pred, labels=valid)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=valid, yticklabels=valid, ax=ax)
        ax.set_xlabel("Prediksi", fontsize=11, fontweight="bold")
        ax.set_ylabel("Aktual", fontsize=11, fontweight="bold")
        ax.set_title("Confusion Matrix - Klasifikasi Tema Laporan Posyandu\n(Zero-Shot Learning + LLM)",
                     fontsize=12, fontweight="bold")
        plt.xticks(rotation=20, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        path = save_path or OUTPUT_CONFUSION_MATRIX
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Confusion matrix: {path}")

    def plot_metrics_bar(self, y_true, y_pred, save_path=None):
        """Buat bar chart metrik per kelas."""
        valid = [l for l in self.labels if l in set(y_true) | set(y_pred)]
        p = precision_score(y_true, y_pred, labels=valid, average=None, zero_division=0)
        r = recall_score(y_true, y_pred, labels=valid, average=None, zero_division=0)
        f = f1_score(y_true, y_pred, labels=valid, average=None, zero_division=0)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for idx, (vals, name, color) in enumerate([
            (p, "Precision", "#2196F3"), (r, "Recall", "#4CAF50"), (f, "F1-Score", "#FF9800")
        ]):
            ax = axes[idx]
            bars = ax.barh(valid, vals, color=color, alpha=0.8)
            ax.set_xlim(0, 1.1)
            ax.set_title(name, fontsize=12, fontweight="bold")
            for bar, v in zip(bars, vals):
                ax.text(v + 0.02, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center", fontsize=9)

        plt.suptitle("Metrik per Tema - Zero-Shot Learning", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = save_path or OUTPUT_CLASSIFICATION_REPORT
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Grafik metrik: {path}")

    def generate_report(self, y_true, y_pred, model_name="Unknown", save_path=None) -> str:
        """Generate laporan evaluasi teks."""
        m = self.calculate_metrics(y_true, y_pred)
        valid = [l for l in self.labels if l in set(y_true) | set(y_pred)]

        lines = [
            "=" * 70,
            "LAPORAN EVALUASI KLASIFIKASI TEMA PELAYANAN KIA",
            "PADA TEKS LAPORAN POSYANDU",
            "Metode: Zero-Shot Learning + Large Language Model (LLM)",
            "=" * 70, "",
            f"Model         : {model_name}",
            f"Total Sampel  : {m['total_samples']}",
            f"Prediksi Benar: {m['correct']}",
            f"Accuracy      : {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)", "",
            f"Precision (weighted): {m['precision_weighted']:.4f}",
            f"Recall (weighted)   : {m['recall_weighted']:.4f}",
            f"F1-Score (weighted) : {m['f1_weighted']:.4f}", "",
            f"Precision (macro)   : {m['precision_macro']:.4f}",
            f"Recall (macro)      : {m['recall_macro']:.4f}",
            f"F1-Score (macro)    : {m['f1_macro']:.4f}", "",
            "-" * 70,
            "CLASSIFICATION REPORT:", "",
            classification_report(y_true, y_pred, labels=valid, target_names=valid, zero_division=0, digits=4),
            "-" * 70,
            "ANALISIS KESALAHAN:", "",
        ]

        errors = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
        if errors:
            lines.append(f"Total kesalahan: {len(errors)}/{len(y_true)} ({len(errors)/len(y_true)*100:.1f}%)")
            pairs = {}
            for a, p in errors:
                pairs[f"{a} -> {p}"] = pairs.get(f"{a} -> {p}", 0) + 1
            for pair, cnt in sorted(pairs.items(), key=lambda x: -x[1]):
                lines.append(f"  [{cnt}x] {pair}")
        else:
            lines.append("Tidak ada kesalahan! Akurasi 100%")

        lines.append("\n" + "=" * 70)
        report = "\n".join(lines)

        path = save_path or OUTPUT_REPORT
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[INFO] Laporan: {path}")
        return report

    def evaluate_all(self, y_true, y_pred, model_name="Unknown") -> dict:
        """Jalankan semua evaluasi."""
        print("\n" + "=" * 60)
        print("EVALUASI HASIL KLASIFIKASI")
        print("=" * 60)

        m = self.calculate_metrics(y_true, y_pred)
        print(f"\n[RESULT] Accuracy: {m['accuracy']*100:.2f}%")
        print(f"[RESULT] F1 (weighted): {m['f1_weighted']:.4f}")

        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_metrics_bar(y_true, y_pred)
        report = self.generate_report(y_true, y_pred, model_name)
        print("\n" + report)
        return m

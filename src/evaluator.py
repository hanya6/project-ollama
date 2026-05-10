"""
Modul Evaluasi Hasil Klasifikasi
Menghitung metrik evaluasi dan menghasilkan visualisasi
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tabulate import tabulate

from config import (
    TEMA_LABELS,
    RESULTS_DIR,
    OUTPUT_CONFUSION_MATRIX,
    OUTPUT_CLASSIFICATION_REPORT,
    OUTPUT_REPORT,
)


class ClassificationEvaluator:
    """
    Evaluator untuk menghitung dan memvisualisasikan
    performa klasifikasi tema laporan posyandu.
    """
    
    def __init__(self, labels: list = None):
        """
        Inisialisasi evaluator.
        
        Args:
            labels: Daftar label tema (default dari config)
        """
        self.labels = labels or TEMA_LABELS
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
    def calculate_metrics(
        self, 
        y_true: list, 
        y_pred: list,
        average: str = "weighted"
    ) -> dict:
        """
        Menghitung metrik evaluasi klasifikasi.
        
        Args:
            y_true: Label aktual
            y_pred: Label prediksi
            average: Metode averaging ('weighted', 'macro', 'micro')
            
        Returns:
            Dictionary berisi metrik evaluasi
        """
        # Filter hanya label yang ada di data
        unique_labels = sorted(set(y_true) | set(y_pred))
        valid_labels = [l for l in self.labels if l in unique_labels]
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(
                y_true, y_pred, labels=valid_labels, 
                average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, labels=valid_labels, 
                average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, labels=valid_labels, 
                average="weighted", zero_division=0
            ),
            "precision_macro": precision_score(
                y_true, y_pred, labels=valid_labels, 
                average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, labels=valid_labels, 
                average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, labels=valid_labels, 
                average="macro", zero_division=0
            ),
            "total_samples": len(y_true),
            "correct_predictions": sum(1 for t, p in zip(y_true, y_pred) if t == p),
            "labels_used": valid_labels,
        }
        
        return metrics
    
    def get_classification_report(
        self, 
        y_true: list, 
        y_pred: list
    ) -> str:
        """
        Mendapatkan classification report detail.
        
        Args:
            y_true: Label aktual
            y_pred: Label prediksi
            
        Returns:
            String classification report
        """
        unique_labels = sorted(set(y_true) | set(y_pred))
        valid_labels = [l for l in self.labels if l in unique_labels]
        
        report = classification_report(
            y_true, y_pred,
            labels=valid_labels,
            target_names=valid_labels,
            zero_division=0,
            digits=4
        )
        
        return report
    
    def get_per_class_metrics(
        self, 
        y_true: list, 
        y_pred: list
    ) -> pd.DataFrame:
        """
        Mendapatkan metrik per kelas dalam bentuk DataFrame.
        
        Args:
            y_true: Label aktual
            y_pred: Label prediksi
            
        Returns:
            DataFrame metrik per kelas
        """
        unique_labels = sorted(set(y_true) | set(y_pred))
        valid_labels = [l for l in self.labels if l in unique_labels]
        
        precision_per_class = precision_score(
            y_true, y_pred, labels=valid_labels, 
            average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, labels=valid_labels, 
            average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, labels=valid_labels, 
            average=None, zero_division=0
        )
        
        # Hitung support (jumlah sampel per kelas)
        support = []
        for label in valid_labels:
            support.append(sum(1 for t in y_true if t == label))
        
        df_metrics = pd.DataFrame({
            "Tema": valid_labels,
            "Precision": precision_per_class,
            "Recall": recall_per_class,
            "F1-Score": f1_per_class,
            "Support": support,
        })
        
        return df_metrics
    
    def plot_confusion_matrix(
        self, 
        y_true: list, 
        y_pred: list,
        save_path: str = None,
        figsize: tuple = (12, 10)
    ):
        """
        Membuat visualisasi confusion matrix.
        
        Args:
            y_true: Label aktual
            y_pred: Label prediksi
            save_path: Path untuk menyimpan gambar
            figsize: Ukuran figure
        """
        unique_labels = sorted(set(y_true) | set(y_pred))
        valid_labels = [l for l in self.labels if l in unique_labels]
        
        # Buat short labels untuk display
        short_labels = []
        for label in valid_labels:
            if "(" in label:
                # Ambil singkatan dalam kurung
                short = label.split("(")[1].replace(")", "").strip()
                short_labels.append(short)
            elif "&" in label:
                short_labels.append(label.split("&")[0].strip()[:15])
            else:
                short_labels.append(label[:20])
        
        cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        
        ax.set_xlabel("Prediksi", fontsize=12, fontweight="bold")
        ax.set_ylabel("Aktual", fontsize=12, fontweight="bold")
        ax.set_title(
            "Confusion Matrix\nKlasifikasi Tema Laporan Posyandu (Zero-Shot Learning)",
            fontsize=14,
            fontweight="bold",
            pad=20
        )
        
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Simpan
        save_to = save_path or OUTPUT_CONFUSION_MATRIX
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[INFO] Confusion matrix disimpan: {save_to}")
    
    def plot_classification_metrics(
        self, 
        y_true: list, 
        y_pred: list,
        save_path: str = None,
        figsize: tuple = (14, 8)
    ):
        """
        Membuat visualisasi metrik per kelas (bar chart).
        
        Args:
            y_true: Label aktual
            y_pred: Label prediksi
            save_path: Path untuk menyimpan gambar
            figsize: Ukuran figure
        """
        df_metrics = self.get_per_class_metrics(y_true, y_pred)
        
        # Buat short labels
        short_labels = []
        for label in df_metrics["Tema"]:
            if "(" in label:
                short = label.split("(")[1].replace(")", "").strip()
                short_labels.append(short)
            elif "&" in label:
                short_labels.append("Pertumbuhan")
            else:
                short_labels.append(label[:15])
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics_to_plot = ["Precision", "Recall", "F1-Score"]
        colors = ["#2196F3", "#4CAF50", "#FF9800"]
        
        for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            ax = axes[idx]
            bars = ax.barh(short_labels, df_metrics[metric], color=color, alpha=0.8)
            ax.set_xlim(0, 1.1)
            ax.set_xlabel(metric, fontweight="bold")
            ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)
            
            # Tambah label nilai
            for bar, val in zip(bars, df_metrics[metric]):
                ax.text(
                    val + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=9
                )
            
            ax.set_title(metric, fontsize=13, fontweight="bold")
        
        plt.suptitle(
            "Metrik Klasifikasi per Tema\n(Zero-Shot Learning - Laporan Posyandu)",
            fontsize=14,
            fontweight="bold",
            y=1.02
        )
        plt.tight_layout()
        
        # Simpan
        save_to = save_path or OUTPUT_CLASSIFICATION_REPORT
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[INFO] Grafik metrik disimpan: {save_to}")
    
    def plot_prediction_distribution(
        self,
        y_pred: list,
        save_path: str = None,
        figsize: tuple = (10, 6)
    ):
        """
        Membuat visualisasi distribusi hasil prediksi.
        
        Args:
            y_pred: Label prediksi
            save_path: Path untuk menyimpan gambar
            figsize: Ukuran figure
        """
        pred_counts = pd.Series(y_pred).value_counts()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = sns.color_palette("Set2", len(pred_counts))
        bars = ax.bar(range(len(pred_counts)), pred_counts.values, color=colors)
        
        # Short labels
        short_labels = []
        for label in pred_counts.index:
            if "(" in label:
                short = label.split("(")[1].replace(")", "").strip()
                short_labels.append(short)
            elif "&" in label:
                short_labels.append("Pertumbuhan")
            else:
                short_labels.append(label[:15])
        
        ax.set_xticks(range(len(pred_counts)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right")
        ax.set_ylabel("Jumlah", fontweight="bold")
        ax.set_title(
            "Distribusi Hasil Klasifikasi Tema\nLaporan Posyandu",
            fontsize=13,
            fontweight="bold"
        )
        
        # Tambah label jumlah di atas bar
        for bar, val in zip(bars, pred_counts.values):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontweight="bold"
            )
        
        plt.tight_layout()
        
        save_to = save_path or os.path.join(self.results_dir, "distribusi_prediksi.png")
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[INFO] Distribusi prediksi disimpan: {save_to}")
    
    def generate_full_report(
        self, 
        y_true: list, 
        y_pred: list,
        model_name: str = "Unknown",
        save_path: str = None
    ) -> str:
        """
        Menghasilkan laporan evaluasi lengkap dalam bentuk teks.
        
        Args:
            y_true: Label aktual
            y_pred: Label prediksi
            model_name: Nama model yang digunakan
            save_path: Path untuk menyimpan laporan
            
        Returns:
            String laporan evaluasi
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        df_per_class = self.get_per_class_metrics(y_true, y_pred)
        
        report = []
        report.append("=" * 70)
        report.append("LAPORAN EVALUASI KLASIFIKASI TEMA LAPORAN POSYANDU")
        report.append("Metode: Zero-Shot Learning dengan Large Language Model")
        report.append("=" * 70)
        report.append("")
        report.append(f"Model LLM        : {model_name}")
        report.append(f"Total Sampel     : {metrics['total_samples']}")
        report.append(f"Prediksi Benar   : {metrics['correct_predictions']}")
        report.append(f"Jumlah Kelas     : {len(metrics['labels_used'])}")
        report.append("")
        report.append("-" * 70)
        report.append("METRIK KESELURUHAN")
        report.append("-" * 70)
        report.append("")
        report.append(f"  Accuracy              : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        report.append("")
        report.append(f"  Precision (weighted)  : {metrics['precision_weighted']:.4f}")
        report.append(f"  Recall (weighted)     : {metrics['recall_weighted']:.4f}")
        report.append(f"  F1-Score (weighted)   : {metrics['f1_weighted']:.4f}")
        report.append("")
        report.append(f"  Precision (macro)     : {metrics['precision_macro']:.4f}")
        report.append(f"  Recall (macro)        : {metrics['recall_macro']:.4f}")
        report.append(f"  F1-Score (macro)      : {metrics['f1_macro']:.4f}")
        report.append("")
        report.append("-" * 70)
        report.append("METRIK PER KELAS")
        report.append("-" * 70)
        report.append("")
        
        # Format tabel per kelas
        table_data = []
        for _, row in df_per_class.iterrows():
            table_data.append([
                row["Tema"][:35],
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1-Score']:.4f}",
                int(row["Support"]),
            ])
        
        table_str = tabulate(
            table_data,
            headers=["Tema", "Precision", "Recall", "F1-Score", "Support"],
            tablefmt="grid",
            stralign="left",
            numalign="center",
        )
        report.append(table_str)
        
        report.append("")
        report.append("-" * 70)
        report.append("CLASSIFICATION REPORT (sklearn)")
        report.append("-" * 70)
        report.append("")
        report.append(self.get_classification_report(y_true, y_pred))
        
        report.append("")
        report.append("=" * 70)
        report.append("ANALISIS KESALAHAN")
        report.append("=" * 70)
        report.append("")
        
        # Analisis kesalahan
        errors = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
        if errors:
            report.append(f"Total kesalahan: {len(errors)}/{len(y_true)}")
            report.append("")
            error_pairs = {}
            for actual, predicted in errors:
                key = f"{actual} -> {predicted}"
                error_pairs[key] = error_pairs.get(key, 0) + 1
            
            report.append("Pola kesalahan (Aktual -> Prediksi):")
            for pair, count in sorted(error_pairs.items(), key=lambda x: -x[1]):
                report.append(f"  [{count}x] {pair}")
        else:
            report.append("Tidak ada kesalahan! Akurasi 100%")
        
        report.append("")
        report.append("=" * 70)
        
        # Gabungkan report
        full_report = "\n".join(report)
        
        # Simpan
        save_to = save_path or OUTPUT_REPORT
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, "w", encoding="utf-8") as f:
            f.write(full_report)
        
        print(f"[INFO] Laporan evaluasi disimpan: {save_to}")
        
        return full_report
    
    def evaluate_all(
        self, 
        y_true: list, 
        y_pred: list, 
        model_name: str = "Unknown"
    ) -> dict:
        """
        Jalankan semua evaluasi sekaligus (metrik, visualisasi, laporan).
        
        Args:
            y_true: Label aktual
            y_pred: Label prediksi
            model_name: Nama model yang digunakan
            
        Returns:
            Dictionary berisi semua metrik
        """
        print("\n" + "=" * 60)
        print("EVALUASI HASIL KLASIFIKASI")
        print("=" * 60 + "\n")
        
        # 1. Hitung metrik
        metrics = self.calculate_metrics(y_true, y_pred)
        print(f"[INFO] Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"[INFO] F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        
        # 2. Buat confusion matrix
        print("\n[INFO] Membuat confusion matrix...")
        self.plot_confusion_matrix(y_true, y_pred)
        
        # 3. Buat grafik metrik per kelas
        print("[INFO] Membuat grafik metrik per kelas...")
        self.plot_classification_metrics(y_true, y_pred)
        
        # 4. Buat distribusi prediksi
        print("[INFO] Membuat grafik distribusi...")
        self.plot_prediction_distribution(y_pred)
        
        # 5. Generate laporan teks
        print("[INFO] Membuat laporan evaluasi...")
        report = self.generate_full_report(y_true, y_pred, model_name)
        
        print("\n" + report)
        
        return metrics

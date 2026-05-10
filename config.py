"""
Konfigurasi Project
KLASIFIKASI TEMA PELAYANAN KESEHATAN IBU DAN ANAK
PADA TEKS LAPORAN POSYANDU MENGGUNAKAN LARGE LANGUAGE MODELS (LLM)
DAN ZERO-SHOT LEARNING
"""

import os

# ============================================================
# KONFIGURASI OLLAMA
# ============================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model LLM: "llama3", "mistral", "gemma:7b", dll
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Parameter generasi teks
GENERATION_CONFIG = {
    "temperature": 0.1,       # Rendah untuk konsistensi klasifikasi
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 100,       # Maksimal token output
    "repeat_penalty": 1.1,
}

# ============================================================
# KONFIGURASI DATA
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GIZI_BALITA_PATH = os.path.join(DATA_DIR, "gizi_balita.csv")
IMUNISASI_BAYI_PATH = os.path.join(DATA_DIR, "imunisasi_bayi.csv")
KESEHATAN_BALITA_PATH = os.path.join(DATA_DIR, "kesehatan_balita.csv")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
GENERATED_DATA_PATH = os.path.join(DATA_DIR, "generated_laporan.csv")

# ============================================================
# KONFIGURASI KLASIFIKASI - 3 TEMA PELAYANAN KIA
# ============================================================

TEMA_LABELS = [
    "Pemantauan Gizi Balita",
    "Imunisasi Bayi",
    "Pemeriksaan Kesehatan Balita",
]

TEMA_DESCRIPTIONS = {
    "Pemantauan Gizi Balita": (
        "Kegiatan penimbangan berat badan, pengukuran tinggi badan dan lingkar kepala balita. "
        "Pemantauan status gizi (gizi baik, gizi kurang, stunting) dan pencatatan ASI eksklusif. "
        "Laporan ini HANYA membahas pertumbuhan dan gizi, TANPA menyebutkan status imunisasi."
    ),
    "Imunisasi Bayi": (
        "Pemberian vaksin/imunisasi pada bayi meliputi BCG, DPT-HB-Hib, Polio, dan Campak. "
        "Pencatatan status kelengkapan imunisasi dan pemantauan KIPI (demam ringan). "
        "Laporan ini HANYA membahas vaksinasi, TANPA pengukuran berat badan atau status gizi."
    ),
    "Pemeriksaan Kesehatan Balita": (
        "Pemeriksaan kesehatan TERPADU yang MENGGABUNGKAN informasi pertumbuhan/gizi "
        "DAN status kelengkapan imunisasi dalam SATU laporan. "
        "Ciri khas: menyebutkan KEDUA informasi yaitu status gizi DAN status imunisasi sekaligus."
    ),
}

# ============================================================
# KONFIGURASI OUTPUT
# ============================================================

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "hasil_klasifikasi.csv")
OUTPUT_REPORT = os.path.join(RESULTS_DIR, "laporan_evaluasi.txt")
OUTPUT_CONFUSION_MATRIX = os.path.join(RESULTS_DIR, "confusion_matrix.png")
OUTPUT_CLASSIFICATION_REPORT = os.path.join(RESULTS_DIR, "classification_report.png")

# ============================================================
# KONFIGURASI LAINNYA
# ============================================================

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

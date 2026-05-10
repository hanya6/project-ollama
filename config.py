"""
Konfigurasi Project
Klasifikasi Tema Pelayanan Kesehatan Ibu dan Anak
pada Teks Laporan Posyandu menggunakan LLM dan Zero-Shot Learning
"""

import os

# ============================================================
# KONFIGURASI OLLAMA
# ============================================================

# URL endpoint Ollama API
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model LLM yang digunakan
# Opsi: "llama3", "mistral", "llama3:8b", "gemma:7b", dll
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Parameter generasi teks
GENERATION_CONFIG = {
    "temperature": 0.1,       # Rendah untuk konsistensi klasifikasi
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 100,       # Maksimal token yang dihasilkan
    "repeat_penalty": 1.1,
}

# ============================================================
# KONFIGURASI DATA
# ============================================================

# Path data
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GIZI_BALITA_PATH = os.path.join(DATA_DIR, "gizi_balita.csv")
IMUNISASI_BAYI_PATH = os.path.join(DATA_DIR, "imunisasi_bayi.csv")
KESEHATAN_BALITA_PATH = os.path.join(DATA_DIR, "kesehatan_balita.csv")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

# Path data generated (teks laporan hasil generate)
GENERATED_DATA_PATH = os.path.join(DATA_DIR, "generated_laporan.csv")

# ============================================================
# KONFIGURASI KLASIFIKASI
# ============================================================

# Daftar tema/label klasifikasi pelayanan KIA di Posyandu
TEMA_LABELS = [
    "Pemantauan Gizi Balita",
    "Imunisasi Bayi",
    "Pemeriksaan Kesehatan Balita",
]

# Deskripsi setiap tema untuk membantu Zero-Shot Learning
TEMA_DESCRIPTIONS = {
    "Pemantauan Gizi Balita": (
        "Kegiatan penimbangan berat badan, pengukuran tinggi badan, dan lingkar kepala balita. "
        "Pemantauan status gizi meliputi gizi baik, gizi kurang, dan stunting. "
        "Termasuk pencatatan ASI eksklusif, kenaikan atau penurunan berat badan, "
        "serta rujukan untuk balita dengan status gizi buruk atau stunting."
    ),
    "Imunisasi Bayi": (
        "Pemberian vaksin/imunisasi pada bayi dan balita meliputi BCG, DPT-HB-Hib, "
        "Polio, dan Campak. Pencatatan status kelengkapan imunisasi (lengkap/belum lengkap), "
        "pemantauan Kejadian Ikutan Pasca Imunisasi (KIPI) seperti demam ringan, "
        "serta pelayanan oleh bidan dan kader posyandu."
    ),
    "Pemeriksaan Kesehatan Balita": (
        "Pemeriksaan kesehatan umum balita yang mencakup pemantauan pertumbuhan "
        "(berat badan, tinggi badan, lingkar kepala), status gizi, kelengkapan imunisasi, "
        "pemberian ASI eksklusif, serta catatan perkembangan dan kesehatan secara menyeluruh. "
        "Merupakan pelayanan kesehatan terpadu untuk balita di posyandu."
    ),
}

# Mapping file CSV ke tema
FILE_TEMA_MAPPING = {
    "gizi_balita.csv": "Pemantauan Gizi Balita",
    "imunisasi_bayi.csv": "Imunisasi Bayi",
    "kesehatan_balita.csv": "Pemeriksaan Kesehatan Balita",
}

# ============================================================
# KONFIGURASI OUTPUT
# ============================================================

# Direktori hasil
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Format output
OUTPUT_CSV = os.path.join(RESULTS_DIR, "hasil_klasifikasi.csv")
OUTPUT_REPORT = os.path.join(RESULTS_DIR, "laporan_evaluasi.txt")
OUTPUT_CONFUSION_MATRIX = os.path.join(RESULTS_DIR, "confusion_matrix.png")
OUTPUT_CLASSIFICATION_REPORT = os.path.join(RESULTS_DIR, "classification_report.png")

# ============================================================
# KONFIGURASI LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

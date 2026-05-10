"""
Konfigurasi Project Klasifikasi Tema Pelayanan KIA
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
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_laporan.csv")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

# ============================================================
# KONFIGURASI KLASIFIKASI
# ============================================================

# Daftar tema/label klasifikasi
TEMA_LABELS = [
    "Pemeriksaan Kehamilan (ANC)",
    "Imunisasi",
    "Penimbangan & Pemantauan Pertumbuhan",
    "Pemberian Makanan Tambahan (PMT)",
    "Penyuluhan Kesehatan",
    "Pelayanan KB",
    "Pemeriksaan Nifas",
    "Deteksi Dini Tumbuh Kembang",
]

# Deskripsi setiap tema untuk membantu Zero-Shot Learning
TEMA_DESCRIPTIONS = {
    "Pemeriksaan Kehamilan (ANC)": (
        "Kegiatan pemeriksaan rutin ibu hamil meliputi pemeriksaan tekanan darah, "
        "berat badan, tinggi fundus uteri, denyut jantung janin, pemberian tablet Fe, "
        "pemeriksaan laboratorium, USG, dan konseling kehamilan."
    ),
    "Imunisasi": (
        "Pemberian vaksin/imunisasi pada bayi dan balita termasuk BCG, DPT-HB-Hib, "
        "Polio (OPV/IPV), Campak/MR, Hepatitis B, dan imunisasi lanjutan. "
        "Termasuk pencatatan status imunisasi dan KIPI."
    ),
    "Penimbangan & Pemantauan Pertumbuhan": (
        "Kegiatan penimbangan berat badan balita, pengukuran tinggi/panjang badan, "
        "pengukuran lingkar kepala, pencatatan di KMS/buku KIA, "
        "pemantauan status gizi (gizi baik, kurang, buruk, lebih)."
    ),
    "Pemberian Makanan Tambahan (PMT)": (
        "Pemberian makanan tambahan untuk balita gizi kurang/buruk dan ibu hamil KEK, "
        "termasuk PMT pemulihan, PMT penyuluhan, pemberian biskuit, "
        "susu, bubur, dan makanan bergizi lainnya."
    ),
    "Penyuluhan Kesehatan": (
        "Kegiatan edukasi dan penyuluhan kesehatan kepada ibu dan masyarakat "
        "tentang ASI eksklusif, MPASI, gizi seimbang, PHBS, pencegahan penyakit, "
        "perawatan bayi/balita, dan kesehatan reproduksi."
    ),
    "Pelayanan KB": (
        "Pelayanan keluarga berencana meliputi konseling KB, pemberian pil KB, "
        "suntik KB, pemasangan implant/IUD, kondom, dan pemantauan efek samping "
        "alat kontrasepsi."
    ),
    "Pemeriksaan Nifas": (
        "Kunjungan dan pemeriksaan ibu pasca persalinan (masa nifas), "
        "pemeriksaan involusi uterus, pemantauan lochea, perawatan luka, "
        "konseling menyusui, dan deteksi komplikasi nifas."
    ),
    "Deteksi Dini Tumbuh Kembang": (
        "Kegiatan stimulasi, deteksi, dan intervensi dini tumbuh kembang anak "
        "menggunakan KPSP, pemeriksaan penglihatan, pendengaran, "
        "perkembangan motorik, bahasa, dan sosial-emosional."
    ),
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

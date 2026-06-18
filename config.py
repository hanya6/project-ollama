import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

GIZI_CSV = os.path.join(DATA_DIR, "gizi_balita.csv")
KESEHATAN_CSV = os.path.join(DATA_DIR, "kesehatan_balita.csv")
IMUNISASI_CSV = os.path.join(DATA_DIR, "imunisasi_bayi.csv")

GENERATED_DATA_PATH = os.path.join(RESULTS_DIR, "generated_laporan_posyandu.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "hasil_klasifikasi.csv")

OLLAMA_MODEL = "mistral"

TEMA_LABELS = [
    "Gizi Balita",
    "Kesehatan Balita",
    "Imunisasi Bayi"
]

VERBOSE = True

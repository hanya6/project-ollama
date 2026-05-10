# KLASIFIKASI TEMA PELAYANAN KESEHATAN IBU DAN ANAK PADA TEKS LAPORAN POSYANDU MENGGUNAKAN LARGE LANGUAGE MODELS (LLM) DAN ZERO-SHOT LEARNING

## Deskripsi

Sistem klasifikasi otomatis tema pelayanan kesehatan ibu dan anak (KIA) pada teks laporan Posyandu menggunakan **Large Language Models (LLM)** via **Ollama** dengan pendekatan **Zero-Shot Learning**.

## Tema Klasifikasi (3 Kelas)

| No | Tema | Sumber Data | Jumlah |
|----|------|-------------|--------|
| 1 | Pemantauan Gizi Balita | `gizi_balita.csv` | 300 |
| 2 | Imunisasi Bayi | `imunisasi_bayi.csv` | 300 |
| 3 | Pemeriksaan Kesehatan Balita | `kesehatan_balita.csv` | 300 |

## Alur Kerja

```
Data Tabular (CSV)          ->  Teks Laporan Naratif
  - gizi_balita.csv               (generated_laporan.csv)
  - imunisasi_bayi.csv
  - kesehatan_balita.csv
                                       |
                                       v
                              LLM Zero-Shot Classification
                              (Ollama: llama3/mistral/dll)
                                       |
                                       v
                              Evaluasi & Visualisasi
                              (Accuracy, F1, Confusion Matrix)
```

## Struktur Project

```
project-ollama/
├── main.py                  # Script utama (pipeline)
├── evaluate.py              # Evaluasi & visualisasi
├── config.py                # Konfigurasi
├── requirements.txt         # Dependencies
├── data/
│   ├── gizi_balita.csv      # 300 data gizi balita
│   ├── imunisasi_bayi.csv   # 300 data imunisasi
│   ├── kesehatan_balita.csv # 300 data kesehatan
│   ├── labels.json          # Label tema
│   └── generated_laporan.csv # [generated] teks naratif
├── src/
│   ├── text_generator.py    # Generate teks dari CSV
│   ├── classifier.py        # Zero-shot classifier (Ollama)
│   ├── prompt_templates.py  # Template prompt LLM
│   ├── data_loader.py       # Loading data
│   └── evaluator.py         # Evaluasi metrik
└── results/                 # Output (grafik, laporan)
```

## Instalasi

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Download model
ollama pull llama3

# 3. Jalankan Ollama
ollama serve

# 4. Install Python dependencies
pip install -r requirements.txt
```

## Penggunaan

```bash
# Pipeline lengkap (generate + klasifikasi)
python main.py

# Generate teks laporan saja
python main.py --generate

# Klasifikasi saja (data sudah di-generate)
python main.py --classify

# Dengan model tertentu
python main.py --model mistral

# Dengan Chain-of-Thought reasoning
python main.py --reasoning

# Batasi data (untuk testing cepat)
python main.py --limit 30

# Klasifikasi satu teks
python main.py --single "Hari ini dilakukan penimbangan balita, BB 12 kg, status gizi baik"

# Evaluasi hasil
python evaluate.py
```

## Metodologi Zero-Shot Learning

Model LLM mengklasifikasikan teks laporan posyandu tanpa pernah dilatih pada data berlabel. Pendekatan:
1. **Text Generation** - Data tabular diubah menjadi teks naratif
2. **Prompt Engineering** - Prompt berisi deskripsi setiap tema
3. **LLM Inference** - Model memahami konteks dan memilih tema
4. **Evaluation** - Mengukur akurasi, precision, recall, F1-score

## Referensi

- Ollama: https://ollama.com
- Brown et al. (2020) - "Language Models are Few-Shot Learners"
- Wei et al. (2022) - "Finetuned Language Models Are Zero-Shot Learners"

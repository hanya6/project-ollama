# Klasifikasi Tema Pelayanan Kesehatan Ibu dan Anak pada Teks Laporan Posyandu Menggunakan Large Language Models (LLM) dan Zero-Shot Learning

## Deskripsi Project

Project ini mengimplementasikan sistem klasifikasi otomatis untuk mengidentifikasi tema pelayanan kesehatan ibu dan anak (KIA) pada teks laporan Posyandu. Sistem ini memanfaatkan **Large Language Models (LLM)** melalui **Ollama** dengan pendekatan **Zero-Shot Learning**, dimana model dapat mengklasifikasikan teks ke dalam kategori tema tanpa memerlukan data pelatihan berlabel.

## Latar Belakang

Posyandu (Pos Pelayanan Terpadu) merupakan salah satu bentuk upaya kesehatan bersumber daya masyarakat yang dikelola dari, oleh, untuk, dan bersama masyarakat. Laporan kegiatan Posyandu seringkali berisi berbagai informasi terkait pelayanan kesehatan ibu dan anak yang perlu diklasifikasikan untuk keperluan monitoring dan evaluasi program.

### Tema Klasifikasi

Sistem ini mengklasifikasikan teks laporan ke dalam tema-tema berikut:

1. **Pemeriksaan Kehamilan (Antenatal Care/ANC)** - Pemeriksaan rutin ibu hamil, USG, pemberian tablet Fe
2. **Imunisasi** - Pemberian vaksin pada bayi dan balita (BCG, DPT, Polio, Campak, dll)
3. **Penimbangan & Pemantauan Pertumbuhan** - Penimbangan balita, pengukuran tinggi badan, KMS
4. **Pemberian Makanan Tambahan (PMT)** - Pemberian PMT untuk balita gizi kurang/buruk
5. **Penyuluhan Kesehatan** - Edukasi kesehatan ibu dan anak, KB, ASI eksklusif
6. **Pelayanan KB (Keluarga Berencana)** - Pelayanan kontrasepsi, konseling KB
7. **Pemeriksaan Nifas** - Kunjungan nifas, pemeriksaan pasca persalinan
8. **Deteksi Dini Tumbuh Kembang** - Stimulasi, deteksi, dan intervensi dini tumbuh kembang

## Arsitektur Sistem

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Data Laporan       │────▶│  Zero-Shot Classifier │────▶│  Hasil Klasifikasi  │
│  Posyandu (Teks)    │     │  (Ollama LLM)         │     │  + Evaluasi         │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

## Teknologi yang Digunakan

- **Python 3.10+**
- **Ollama** - Platform untuk menjalankan LLM secara lokal
- **LLM Models** - llama3, mistral, atau model lainnya via Ollama
- **Pandas** - Manipulasi dan analisis data
- **Scikit-learn** - Evaluasi metrik klasifikasi
- **Matplotlib & Seaborn** - Visualisasi hasil

## Struktur Project

```
project-ollama/
├── README.md                   # Dokumentasi project
├── requirements.txt            # Dependencies Python
├── config.py                   # Konfigurasi project
├── data/
│   ├── sample_laporan.csv      # Data sampel laporan posyandu
│   └── labels.json             # Daftar label/tema klasifikasi
├── src/
│   ├── __init__.py
│   ├── classifier.py           # Modul klasifikasi zero-shot
│   ├── data_loader.py          # Modul loading data
│   ├── prompt_templates.py     # Template prompt untuk LLM
│   └── evaluator.py            # Modul evaluasi hasil
├── main.py                     # Script utama
├── evaluate.py                 # Script evaluasi & visualisasi
└── results/                    # Folder output hasil
    └── .gitkeep
```

## Instalasi & Penggunaan

### Prasyarat

1. **Install Ollama**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Download Model LLM**
   ```bash
   ollama pull llama3
   # atau
   ollama pull mistral
   ```

3. **Install Dependencies Python**
   ```bash
   pip install -r requirements.txt
   ```

### Menjalankan Klasifikasi

```bash
# Jalankan klasifikasi pada data sampel
python main.py

# Jalankan dengan model tertentu
python main.py --model llama3

# Jalankan evaluasi dan visualisasi
python evaluate.py
```

## Metodologi Zero-Shot Learning

Zero-Shot Learning memungkinkan model untuk mengklasifikasikan teks ke dalam kategori yang belum pernah dilihat selama pelatihan. Pendekatan ini menggunakan:

1. **Prompt Engineering** - Merancang prompt yang mendeskripsikan tugas klasifikasi
2. **Label Description** - Memberikan deskripsi detail setiap kategori/tema
3. **LLM Reasoning** - Memanfaatkan kemampuan pemahaman bahasa dari LLM

### Contoh Prompt Zero-Shot

```
Klasifikasikan teks laporan posyandu berikut ke dalam salah satu tema:
- Pemeriksaan Kehamilan (ANC)
- Imunisasi
- Penimbangan & Pemantauan Pertumbuhan
- Pemberian Makanan Tambahan (PMT)
- Penyuluhan Kesehatan
- Pelayanan KB
- Pemeriksaan Nifas
- Deteksi Dini Tumbuh Kembang

Teks: "Hari ini dilakukan penimbangan balita sebanyak 45 anak..."
Tema:
```

## Hasil & Evaluasi

Evaluasi dilakukan menggunakan metrik:
- **Accuracy** - Proporsi prediksi yang benar
- **Precision** - Ketepatan prediksi per kelas
- **Recall** - Kelengkapan prediksi per kelas
- **F1-Score** - Harmonic mean dari precision dan recall
- **Confusion Matrix** - Visualisasi distribusi prediksi

## Lisensi

Project ini dibuat untuk keperluan penelitian dan edukasi.

## Referensi

- Ollama Documentation: https://ollama.com
- Zero-Shot Learning: Wei et al. (2022) - "Finetuned Language Models Are Zero-Shot Learners"
- Pedoman Pelaksanaan Posyandu - Kementerian Kesehatan RI

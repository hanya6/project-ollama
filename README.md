# Klasifikasi Tema Pelayanan Kesehatan Ibu dan Anak pada Teks Laporan Posyandu Menggunakan Large Language Models (LLM) dan Zero-Shot Learning

## Deskripsi Project

Project ini mengimplementasikan sistem klasifikasi otomatis untuk mengidentifikasi tema pelayanan kesehatan ibu dan anak (KIA) pada teks laporan Posyandu. Sistem ini memanfaatkan **Large Language Models (LLM)** melalui **Ollama** dengan pendekatan **Zero-Shot Learning**, dimana model dapat mengklasifikasikan teks ke dalam kategori tema tanpa memerlukan data pelatihan berlabel.

## Latar Belakang

Posyandu (Pos Pelayanan Terpadu) merupakan salah satu bentuk upaya kesehatan bersumber daya masyarakat yang dikelola dari, oleh, untuk, dan bersama masyarakat. Laporan kegiatan Posyandu seringkali berisi berbagai informasi terkait pelayanan kesehatan ibu dan anak yang perlu diklasifikasikan untuk keperluan monitoring dan evaluasi program.

## Tema Klasifikasi

Sistem ini mengklasifikasikan teks laporan ke dalam 3 tema pelayanan KIA:

| No | Tema | Kode | Sumber Data |
|----|------|------|-------------|
| 1 | **Pemantauan Gizi Balita** | GIZI | `gizi_balita.csv` (300 data) |
| 2 | **Imunisasi Bayi** | IMN | `imunisasi_bayi.csv` (300 data) |
| 3 | **Pemeriksaan Kesehatan Balita** | KES | `kesehatan_balita.csv` (300 data) |

### Detail Tema:

1. **Pemantauan Gizi Balita** - Penimbangan berat badan, pengukuran tinggi badan & lingkar kepala, pemantauan status gizi (gizi baik, gizi kurang, stunting), pencatatan ASI eksklusif.

2. **Imunisasi Bayi** - Pemberian vaksin BCG, DPT-HB-Hib, Polio, Campak. Pencatatan status kelengkapan imunisasi dan pemantauan KIPI.

3. **Pemeriksaan Kesehatan Balita** - Pemeriksaan kesehatan terpadu yang mencakup pertumbuhan, status gizi, imunisasi, dan perkembangan secara menyeluruh.

## Arsitektur & Alur Kerja

```
┌────────────────────────┐
│  Data Tabular (CSV)    │
│  - gizi_balita.csv     │
│  - imunisasi_bayi.csv  │
│  - kesehatan_balita.csv│
└──────────┬─────────────┘
           │ Step 1: Text Generation
           ▼
┌────────────────────────┐
│  Teks Laporan Naratif  │
│  (generated_laporan.csv)│
└──────────┬─────────────┘
           │ Step 2: Zero-Shot Classification
           ▼
┌────────────────────────┐
│  LLM via Ollama        │
│  (llama3/mistral/dll)  │
│  + Prompt Engineering  │
└──────────┬─────────────┘
           │ Step 3: Evaluation
           ▼
┌────────────────────────┐
│  Hasil & Evaluasi      │
│  - Accuracy, F1-Score  │
│  - Confusion Matrix    │
│  - Classification Report│
└────────────────────────┘
```

## Metodologi Zero-Shot Learning

Zero-Shot Learning memungkinkan model untuk mengklasifikasikan teks ke dalam kategori yang belum pernah dilihat selama pelatihan. Pendekatan ini menggunakan:

1. **Text Generation** - Data tabular diubah menjadi teks naratif yang menyerupai laporan kegiatan posyandu sesungguhnya.
2. **Prompt Engineering** - Merancang prompt yang mendeskripsikan tugas klasifikasi beserta deskripsi setiap tema.
3. **LLM Inference** - Model LLM memahami konteks teks dan memilih tema yang paling sesuai.
4. **Evaluation** - Mengukur performa klasifikasi menggunakan metrik standar.

## Teknologi yang Digunakan

- **Python 3.10+**
- **Ollama** - Platform untuk menjalankan LLM secara lokal
- **LLM Models** - llama3, mistral, gemma, atau model lainnya
- **Pandas** - Manipulasi dan analisis data
- **Scikit-learn** - Evaluasi metrik klasifikasi
- **Matplotlib & Seaborn** - Visualisasi hasil
- **tqdm** - Progress bar

## Struktur Project

```
project-ollama/
├── README.md                    # Dokumentasi project
├── requirements.txt             # Dependencies Python
├── config.py                    # Konfigurasi project
├── main.py                      # Script utama (pipeline)
├── evaluate.py                  # Script evaluasi & visualisasi
├── data/
│   ├── gizi_balita.csv          # Data gizi balita (300 record)
│   ├── imunisasi_bayi.csv       # Data imunisasi bayi (300 record)
│   ├── kesehatan_balita.csv     # Data kesehatan balita (300 record)
│   ├── labels.json              # Daftar label/tema klasifikasi
│   └── generated_laporan.csv    # [Generated] Teks laporan naratif
├── src/
│   ├── __init__.py
│   ├── text_generator.py        # Generator teks dari data tabular
│   ├── classifier.py            # Klasifikasi zero-shot via Ollama
│   ├── data_loader.py           # Loading data CSV/JSON
│   ├── prompt_templates.py      # Template prompt untuk LLM
│   └── evaluator.py             # Evaluasi & visualisasi hasil
└── results/                     # Folder output hasil
    ├── hasil_klasifikasi.csv    # [Generated] Hasil klasifikasi
    ├── laporan_evaluasi.txt     # [Generated] Laporan evaluasi
    ├── confusion_matrix.png     # [Generated] Visualisasi
    ├── classification_report.png # [Generated] Grafik metrik
    └── distribusi_prediksi.png  # [Generated] Distribusi
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
   # atau model lain:
   # ollama pull mistral
   # ollama pull gemma:7b
   ```

3. **Jalankan Ollama Server**
   ```bash
   ollama serve
   ```

4. **Install Dependencies Python**
   ```bash
   pip install -r requirements.txt
   ```

### Menjalankan Project

#### Pipeline Lengkap (Recommended)
```bash
python main.py
```
Ini akan menjalankan: Generate Teks → Klasifikasi → Simpan Hasil

#### Step-by-Step
```bash
# Step 1: Generate teks laporan dari data CSV
python main.py --generate

# Step 2: Klasifikasi dengan LLM
python main.py --classify

# Step 3: Evaluasi hasil
python evaluate.py
```

#### Opsi Tambahan
```bash
# Gunakan model tertentu
python main.py --model mistral

# Dengan Chain-of-Thought reasoning
python main.py --reasoning

# Batasi jumlah data (untuk testing)
python main.py --limit 30

# Klasifikasi satu teks
python main.py --single "Hari ini dilakukan penimbangan balita Rizki, BB 12 kg, status gizi baik"

# Batasi jumlah data per tema saat generate
python main.py --n-per-tema 50
```

## Dataset

### gizi_balita.csv (300 record)
| Kolom | Deskripsi |
|-------|-----------|
| ID_Balita | ID unik balita |
| Nama_Balita | Nama balita |
| Umur_Bulan | Usia dalam bulan |
| Jenis_Kelamin | Laki-laki/Perempuan |
| Berat_Badan | Berat badan (kg) |
| Tinggi_Badan | Tinggi badan (cm) |
| Lingkar_Kepala | Lingkar kepala (cm) |
| Status_Gizi | Gizi Baik/Gizi Kurang/Stunting |
| ASI_Eksklusif | Ya/Tidak |
| Tanggal_Pemeriksaan | Tanggal pelayanan |
| Kader | Petugas yang melayani |
| Keterangan | Catatan tambahan |

### imunisasi_bayi.csv (300 record)
| Kolom | Deskripsi |
|-------|-----------|
| ID_Bayi | ID unik bayi |
| Nama_Bayi | Nama bayi |
| Tanggal_Lahir | Tanggal lahir |
| Umur_Bulan | Usia dalam bulan |
| Jenis_Kelamin | Laki-laki/Perempuan |
| Jenis_Imunisasi | BCG/DPT-HB-Hib/Polio/Campak |
| Tanggal_Imunisasi | Tanggal pemberian |
| Status_Imunisasi | Lengkap/Belum Lengkap |
| Petugas | Petugas pelaksana |
| Posyandu | Nama posyandu |
| Keterangan | Kondisi pasca imunisasi |

### kesehatan_balita.csv (300 record)
| Kolom | Deskripsi |
|-------|-----------|
| ID_Balita | ID unik balita |
| Nama_Balita | Nama balita |
| Umur_Bulan | Usia dalam bulan |
| Jenis_Kelamin | Laki-laki/Perempuan |
| Berat_Badan | Berat badan (kg) |
| Tinggi_Badan | Tinggi badan (cm) |
| Lingkar_Kepala | Lingkar kepala (cm) |
| Status_Gizi | Status gizi |
| ASI_Eksklusif | Ya/Tidak |
| Imunisasi_Lengkap | Ya/Tidak |
| Tanggal_Pemeriksaan | Tanggal pelayanan |
| Kader | Petugas yang melayani |
| Catatan | Catatan perkembangan |

## Hasil & Evaluasi

Evaluasi dilakukan menggunakan metrik:
- **Accuracy** - Proporsi prediksi yang benar secara keseluruhan
- **Precision** - Ketepatan prediksi per kelas tema
- **Recall** - Kelengkapan prediksi per kelas tema
- **F1-Score** - Harmonic mean dari precision dan recall
- **Confusion Matrix** - Visualisasi distribusi prediksi vs aktual

## Lisensi

Project ini dibuat untuk keperluan penelitian dan edukasi di bidang Natural Language Processing (NLP) dan kesehatan masyarakat.

## Referensi

- Ollama Documentation: https://ollama.com
- Zero-Shot Learning: Wei et al. (2022) - "Finetuned Language Models Are Zero-Shot Learners"
- Pedoman Pelaksanaan Posyandu - Kementerian Kesehatan RI
- Brown et al. (2020) - "Language Models are Few-Shot Learners" (GPT-3)

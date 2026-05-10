"""
Template Prompt untuk Zero-Shot Classification
menggunakan Large Language Models via Ollama.

Strategi prompt yang diterapkan:
1. Deskripsi tema yang lebih kontrastif (menekankan PERBEDAAN antar tema)
2. Aturan disambiguasi eksplisit untuk kasus yang mirip
3. Chain-of-Thought reasoning untuk meningkatkan akurasi
"""


def get_zero_shot_prompt(teks: str, labels: list, descriptions: dict) -> str:
    """Prompt zero-shot dengan aturan disambiguasi."""
    tema_list = ""
    for i, label in enumerate(labels, 1):
        desc = descriptions.get(label, "")
        tema_list += f"  {i}. {label}: {desc}\n"

    return f"""Kamu adalah asisten ahli kesehatan masyarakat Indonesia, khususnya program Posyandu untuk kesehatan ibu dan anak.

TUGAS: Klasifikasikan teks laporan Posyandu berikut ke dalam SATU tema yang paling sesuai.

DAFTAR TEMA:
{tema_list}
ATURAN PENTING UNTUK MEMBEDAKAN TEMA:
- "Pemantauan Gizi Balita": Fokus HANYA pada penimbangan dan status gizi. TIDAK menyebutkan status imunisasi.
- "Imunisasi Bayi": Fokus pada pemberian vaksin (BCG/DPT/Polio/Campak) dan KIPI.
- "Pemeriksaan Kesehatan Balita": Laporan TERPADU yang menyebutkan KOMBINASI status gizi DAN status imunisasi dalam satu laporan. Jika teks menyebut KEDUA hal tersebut (gizi + imunisasi), pilih tema ini.

TEKS LAPORAN:
"{teks}"

INSTRUKSI:
- Jika teks menyebutkan status gizi DAN status imunisasi sekaligus -> pilih "Pemeriksaan Kesehatan Balita"
- Jika teks HANYA menyebutkan penimbangan/gizi tanpa imunisasi -> pilih "Pemantauan Gizi Balita"
- Jika teks HANYA menyebutkan vaksin/imunisasi -> pilih "Imunisasi Bayi"
- Jawab HANYA dengan nama tema persis seperti di daftar.
- JANGAN beri penjelasan tambahan.

TEMA:"""


def get_zero_shot_prompt_with_reasoning(teks: str, labels: list, descriptions: dict) -> str:
    """Prompt zero-shot dengan Chain-of-Thought reasoning dan disambiguasi."""
    tema_list = ""
    for i, label in enumerate(labels, 1):
        desc = descriptions.get(label, "")
        tema_list += f"  {i}. {label}: {desc}\n"

    return f"""Kamu adalah asisten ahli kesehatan masyarakat Indonesia, khususnya program Posyandu untuk kesehatan ibu dan anak.

TUGAS: Klasifikasikan teks laporan Posyandu berikut ke dalam SATU tema yang paling sesuai.

DAFTAR TEMA:
{tema_list}
ATURAN DISAMBIGUASI:
- Jika teks menyebutkan status gizi DAN status/kelengkapan imunisasi dalam SATU laporan -> "Pemeriksaan Kesehatan Balita"
- Jika teks HANYA fokus pada penimbangan, berat badan, tinggi badan, status gizi, TANPA menyebut imunisasi -> "Pemantauan Gizi Balita"
- Jika teks fokus pada pemberian vaksin (BCG/DPT-HB-Hib/Polio/Campak) dan KIPI -> "Imunisasi Bayi"

TEKS LAPORAN:
"{teks}"

INSTRUKSI:
1. Cek apakah teks menyebutkan informasi imunisasi (vaksin/status imunisasi lengkap/belum lengkap).
2. Cek apakah teks menyebutkan informasi gizi (BB/TB/status gizi/stunting).
3. Jika KEDUANYA ada -> "Pemeriksaan Kesehatan Balita"
4. Jika HANYA gizi -> "Pemantauan Gizi Balita"
5. Jika HANYA imunisasi -> "Imunisasi Bayi"

FORMAT JAWABAN:
Alasan: [sebutkan apakah teks mengandung info gizi saja, imunisasi saja, atau keduanya]
Tema: [nama tema persis seperti di daftar]"""

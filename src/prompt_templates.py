"""
Template Prompt untuk Zero-Shot Classification
menggunakan Large Language Models via Ollama
"""


def get_zero_shot_prompt(teks_laporan: str, tema_labels: list, tema_descriptions: dict) -> str:
    """
    Membuat prompt zero-shot untuk klasifikasi tema laporan posyandu.
    
    Args:
        teks_laporan: Teks laporan posyandu yang akan diklasifikasikan
        tema_labels: Daftar label tema yang tersedia
        tema_descriptions: Deskripsi masing-masing tema
        
    Returns:
        Prompt string yang siap dikirim ke LLM
    """
    # Bangun daftar tema beserta deskripsi
    tema_list = ""
    for i, label in enumerate(tema_labels, 1):
        desc = tema_descriptions.get(label, "")
        tema_list += f"  {i}. {label}: {desc}\n"

    prompt = f"""Kamu adalah asisten ahli di bidang kesehatan masyarakat Indonesia, khususnya dalam program Posyandu (Pos Pelayanan Terpadu) untuk kesehatan ibu dan anak.

TUGAS: Klasifikasikan teks laporan Posyandu berikut ke dalam SATU tema yang paling sesuai.

DAFTAR TEMA YANG TERSEDIA:
{tema_list}
TEKS LAPORAN POSYANDU:
\"{teks_laporan}\"

INSTRUKSI:
- Pilih HANYA SATU tema yang paling sesuai dengan isi laporan di atas.
- Jawab HANYA dengan nama tema persis seperti yang tertulis di daftar.
- JANGAN memberikan penjelasan atau teks tambahan.
- JANGAN mengubah format nama tema.

TEMA YANG PALING SESUAI:"""

    return prompt


def get_zero_shot_prompt_with_reasoning(teks_laporan: str, tema_labels: list, tema_descriptions: dict) -> str:
    """
    Membuat prompt zero-shot dengan Chain-of-Thought reasoning.
    Model diminta menjelaskan alasan sebelum memilih tema.
    
    Args:
        teks_laporan: Teks laporan posyandu yang akan diklasifikasikan
        tema_labels: Daftar label tema yang tersedia
        tema_descriptions: Deskripsi masing-masing tema
        
    Returns:
        Prompt string dengan instruksi reasoning
    """
    tema_list = ""
    for i, label in enumerate(tema_labels, 1):
        desc = tema_descriptions.get(label, "")
        tema_list += f"  {i}. {label}: {desc}\n"

    prompt = f"""Kamu adalah asisten ahli di bidang kesehatan masyarakat Indonesia, khususnya dalam program Posyandu (Pos Pelayanan Terpadu) untuk kesehatan ibu dan anak.

TUGAS: Klasifikasikan teks laporan Posyandu berikut ke dalam SATU tema yang paling sesuai.

DAFTAR TEMA YANG TERSEDIA:
{tema_list}
TEKS LAPORAN POSYANDU:
\"{teks_laporan}\"

INSTRUKSI:
1. Identifikasi kata kunci dan aktivitas utama dalam teks laporan.
2. Cocokkan dengan deskripsi tema yang tersedia.
3. Pilih tema yang paling relevan.

FORMAT JAWABAN:
Alasan: [jelaskan secara singkat mengapa tema ini dipilih]
Tema: [nama tema persis seperti di daftar]"""

    return prompt


def get_multi_label_prompt(teks_laporan: str, tema_labels: list, tema_descriptions: dict) -> str:
    """
    Membuat prompt untuk klasifikasi multi-label (jika satu laporan
    bisa memiliki lebih dari satu tema).
    
    Args:
        teks_laporan: Teks laporan posyandu yang akan diklasifikasikan
        tema_labels: Daftar label tema yang tersedia
        tema_descriptions: Deskripsi masing-masing tema
        
    Returns:
        Prompt string untuk multi-label classification
    """
    tema_list = ""
    for i, label in enumerate(tema_labels, 1):
        desc = tema_descriptions.get(label, "")
        tema_list += f"  {i}. {label}: {desc}\n"

    prompt = f"""Kamu adalah asisten ahli di bidang kesehatan masyarakat Indonesia, khususnya dalam program Posyandu (Pos Pelayanan Terpadu) untuk kesehatan ibu dan anak.

TUGAS: Identifikasi SEMUA tema yang relevan dengan teks laporan Posyandu berikut.

DAFTAR TEMA YANG TERSEDIA:
{tema_list}
TEKS LAPORAN POSYANDU:
\"{teks_laporan}\"

INSTRUKSI:
- Identifikasi semua tema yang relevan (bisa lebih dari satu).
- Urutkan dari yang paling relevan ke kurang relevan.
- Jawab dalam format: Tema1 | Tema2 | Tema3
- Gunakan nama tema persis seperti yang tertulis di daftar.

TEMA YANG RELEVAN:"""

    return prompt

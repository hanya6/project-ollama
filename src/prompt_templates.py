"""
Template Prompt untuk Zero-Shot Classification
menggunakan Large Language Models via Ollama.
"""


def get_zero_shot_prompt(teks: str, labels: list, descriptions: dict) -> str:
    """Prompt zero-shot langsung (tanpa reasoning)."""
    tema_list = ""
    for i, label in enumerate(labels, 1):
        desc = descriptions.get(label, "")
        tema_list += f"  {i}. {label}: {desc}\n"

    return f"""Kamu adalah asisten ahli kesehatan masyarakat Indonesia, khususnya program Posyandu untuk kesehatan ibu dan anak.

TUGAS: Klasifikasikan teks laporan Posyandu berikut ke dalam SATU tema yang paling sesuai.

DAFTAR TEMA:
{tema_list}
TEKS LAPORAN:
"{teks}"

INSTRUKSI:
- Pilih HANYA SATU tema yang paling sesuai.
- Jawab HANYA dengan nama tema persis seperti di daftar.
- JANGAN beri penjelasan tambahan.

TEMA:"""


def get_zero_shot_prompt_with_reasoning(teks: str, labels: list, descriptions: dict) -> str:
    """Prompt zero-shot dengan Chain-of-Thought reasoning."""
    tema_list = ""
    for i, label in enumerate(labels, 1):
        desc = descriptions.get(label, "")
        tema_list += f"  {i}. {label}: {desc}\n"

    return f"""Kamu adalah asisten ahli kesehatan masyarakat Indonesia, khususnya program Posyandu untuk kesehatan ibu dan anak.

TUGAS: Klasifikasikan teks laporan Posyandu berikut ke dalam SATU tema yang paling sesuai.

DAFTAR TEMA:
{tema_list}
TEKS LAPORAN:
"{teks}"

INSTRUKSI:
1. Identifikasi kata kunci dan aktivitas utama dalam teks.
2. Cocokkan dengan deskripsi tema.
3. Pilih tema yang paling relevan.

FORMAT JAWABAN:
Alasan: [penjelasan singkat, maksimal 2 kalimat]
Tema: [nama tema persis seperti di daftar]"""

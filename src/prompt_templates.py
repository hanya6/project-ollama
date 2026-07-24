"""
Template prompt untuk klasifikasi zero-shot menggunakan LLM melalui Ollama.

Prompt dirancang agar model menentukan tema berdasarkan fokus utama laporan,
bukan hanya berdasarkan kemunculan satu kata kunci tertentu.
"""

from typing import Dict, List


def build_theme_list(
    labels: List[str],
    descriptions: Dict[str, str],
) -> str:
    """
    Membentuk daftar tema beserta deskripsinya.

    Args:
        labels:
            Daftar nama tema.

        descriptions:
            Dictionary deskripsi tema dengan format:
            {
                "Nama Tema": "Deskripsi tema"
            }

    Returns:
        Daftar tema dalam bentuk teks.
    """
    theme_lines = []

    for index, label in enumerate(labels, start=1):
        description = descriptions.get(
            label,
            "Tidak ada deskripsi."
        )

        theme_lines.append(
            f"{index}. {label}: {description}"
        )

    return "\n".join(theme_lines)


def get_zero_shot_prompt(
    teks: str,
    labels: List[str],
    descriptions: Dict[str, str],
) -> str:
    """
    Membuat prompt klasifikasi tanpa reasoning.

    Model diminta menentukan tema berdasarkan fokus utama laporan.
    Informasi tambahan tidak otomatis menjadi tema utama.
    """
    tema_list = build_theme_list(
        labels=labels,
        descriptions=descriptions,
    )

    prompt = f"""
Kamu adalah asisten klasifikasi laporan pelayanan Posyandu.

TUGAS:
Kelompokkan teks laporan berikut ke dalam SATU tema yang paling sesuai.

DAFTAR TEMA:
{tema_list}

TEKS LAPORAN:
"{teks}"

ATURAN KLASIFIKASI:
- Tentukan tema berdasarkan tujuan utama laporan.
- Perhatikan tindakan utama yang dilakukan petugas.
- Perhatikan hasil atau temuan utama yang dicatat.
- Informasi tambahan tidak selalu menjadi tema utama.
- Jangan memilih tema hanya karena menemukan satu kata tertentu.
- Jika beberapa topik muncul, pilih topik yang paling dominan.
- Gunakan nama tema persis seperti yang ada pada daftar.
- Jangan memberikan penjelasan tambahan.

FORMAT JAWABAN:
Nama tema
"""

    return prompt.strip()


def get_zero_shot_prompt_with_reasoning(
    teks: str,
    labels: List[str],
    descriptions: Dict[str, str],
) -> str:
    """
    Membuat prompt klasifikasi dengan reasoning singkat.

    Output yang diharapkan:
        Alasan: ...
        Tema: ...
    """
    tema_list = build_theme_list(
        labels=labels,
        descriptions=descriptions,
    )

    prompt = f"""
Kamu adalah asisten klasifikasi laporan pelayanan Posyandu.

TUGAS:
Kelompokkan teks laporan berikut ke dalam SATU tema yang paling sesuai.

DAFTAR TEMA:
{tema_list}

TEKS LAPORAN:
"{teks}"

LANGKAH ANALISIS:
1. Identifikasi tujuan utama kunjungan.
2. Identifikasi tindakan utama petugas.
3. Identifikasi hasil atau temuan utama.
4. Pisahkan informasi utama dari informasi tambahan.
5. Tentukan tema yang paling dominan dalam keseluruhan laporan.

ATURAN:
- Jangan memilih tema hanya berdasarkan satu kata kunci.
- Informasi tambahan tidak otomatis menjadi tema utama.
- Jika beberapa topik disebutkan, pilih fokus yang paling banyak dibahas.
- Gunakan nama tema persis seperti yang tersedia pada daftar.
- Alasan maksimal dua kalimat.

FORMAT JAWABAN:
Alasan: penjelasan singkat
Tema: nama tema
"""

    return prompt.strip()

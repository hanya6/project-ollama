import time
import requests

from config import TEMA_LABELS


class OllamaClassifier:
    def __init__(self, model="mistral", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def check_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def build_prompt(self, teks, with_reasoning=False):
        label_text = "\n".join([f"- {label}" for label in TEMA_LABELS])

        if with_reasoning:
            prompt = f"""
Anda adalah sistem klasifikasi tema laporan posyandu.

Tugas Anda adalah mengklasifikasikan teks laporan berikut ke dalam salah satu tema:

{label_text}

Teks laporan:
\"\"\"{teks}\"\"\"

Berikan jawaban dengan format:
Tema: <salah satu label>
Alasan: <alasan singkat>

Aturan:
- Pilih hanya satu tema.
- Tema harus persis salah satu dari daftar label.
- Jangan membuat tema baru.
"""
        else:
            prompt = f"""
Anda adalah sistem klasifikasi tema laporan posyandu.

Klasifikasikan teks laporan berikut ke dalam salah satu tema:

{label_text}

Teks laporan:
\"\"\"{teks}\"\"\"

Jawab hanya dengan salah satu label berikut:
Gizi Balita
Kesehatan Balita
Imunisasi Bayi
"""

        return prompt.strip()

    def query_ollama(self, prompt):
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9
            }
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()

        return result.get("response", "").strip()

    def normalize_label(self, text):
        text_lower = text.lower()

        if "gizi balita" in text_lower:
            return "Gizi Balita"

        if "kesehatan balita" in text_lower:
            return "Kesehatan Balita"

        if "imunisasi bayi" in text_lower:
            return "Imunisasi Bayi"

        if "imunisasi" in text_lower:
            return "Imunisasi Bayi"

        if "gizi" in text_lower:
            return "Gizi Balita"

        if "kesehatan" in text_lower or "tumbuh kembang" in text_lower:
            return "Kesehatan Balita"

        return "Tidak Diketahui"

    def extract_reason(self, text):
        if "Alasan:" in text:
            return text.split("Alasan:", 1)[1].strip()

        if "alasan:" in text.lower():
            parts = text.lower().split("alasan:", 1)
            if len(parts) > 1:
                return parts[1].strip()

        return ""

    def classify_single(self, teks, with_reasoning=False):
        start = time.time()

        prompt = self.build_prompt(
            teks,
            with_reasoning=with_reasoning
        )

        response = self.query_ollama(prompt)

        tema_prediksi = self.normalize_label(response)
        waktu = round(time.time() - start, 2)

        result = {
            "teks_laporan": teks,
            "tema_prediksi": tema_prediksi,
            "response_llm": response,
            "waktu_proses": waktu
        }

        if with_reasoning:
            result["alasan"] = self.extract_reason(response)

        return result

    def classify_dataframe(self, df, with_reasoning=False, delay=0.5):
        results = []

        total = len(df)

        for idx, row in df.iterrows():
            teks = row["teks_laporan"]

            print(f"[{idx + 1}/{total}] Mengklasifikasikan teks...")

            try:
                result = self.classify_single(
                    teks,
                    with_reasoning=with_reasoning
                )

                result["tema_aktual"] = row.get("tema_aktual", "")

            except Exception as e:
                result = {
                    "teks_laporan": teks,
                    "tema_aktual": row.get("tema_aktual", ""),
                    "tema_prediksi": "ERROR",
                    "response_llm": str(e),
                    "waktu_proses": 0
                }

                if with_reasoning:
                    result["alasan"] = ""

            results.append(result)

            time.sleep(delay)

        import pandas as pd
        return pd.DataFrame(results)

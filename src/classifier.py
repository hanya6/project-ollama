"""
Modul Klasifikasi Zero-Shot menggunakan LLM via Ollama.
"""

import re
import time
from typing import Optional

import requests
from tqdm import tqdm

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GENERATION_CONFIG,
    TEMA_LABELS,
    TEMA_DESCRIPTIONS,
)
from src.prompt_templates import (
    get_zero_shot_prompt,
    get_zero_shot_prompt_with_reasoning,
)


class OllamaClassifier:
    """Klasifikasi Zero-Shot menggunakan LLM via Ollama."""

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or OLLAMA_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.api_url = f"{self.base_url}/api/generate"
        self.tema_labels = TEMA_LABELS
        self.tema_descriptions = TEMA_DESCRIPTIONS
        self.generation_config = GENERATION_CONFIG.copy()

    def check_connection(self) -> bool:
        """Cek koneksi ke Ollama server."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                names = [m["name"] for m in models]
                print(f"[INFO] Terhubung ke Ollama: {self.base_url}")
                print(f"[INFO] Model tersedia: {', '.join(names)}")
                if not any(self.model in n for n in names):
                    print(f"[WARNING] Model '{self.model}' tidak ditemukan!")
                    print(f"[INFO] Jalankan: ollama pull {self.model}")
                    return False
                return True
            return False
        except requests.ConnectionError:
            print(f"[ERROR] Tidak dapat terhubung ke Ollama di {self.base_url}")
            print("[INFO] Pastikan: ollama serve")
            return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Kirim prompt ke Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self.generation_config,
        }
        try:
            resp = requests.post(self.api_url, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            print(f"[ERROR] API error: {resp.status_code}")
            return None
        except requests.Timeout:
            print("[ERROR] Timeout")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def _parse_response(self, response: str) -> str:
        """Ekstrak nama tema dari respons LLM."""
        if not response:
            return "Tidak Terklasifikasi"

        clean = response.strip().strip('"').strip("'").strip()

        # Exact match
        for label in self.tema_labels:
            if label.lower() == clean.lower():
                return label

        # Contains match
        for label in self.tema_labels:
            if label.lower() in clean.lower():
                return label

        # Keyword match
        keywords = {
            "Pemantauan Gizi Balita": [
                "gizi", "penimbangan", "stunting", "gizi kurang",
                "berat badan", "tinggi badan", "antropometri",
            ],
            "Imunisasi Bayi": [
                "imunisasi", "vaksin", "bcg", "dpt", "polio",
                "campak", "kipi", "vaksinasi",
            ],
            "Pemeriksaan Kesehatan Balita": [
                "pemeriksaan kesehatan", "kesehatan balita",
                "kesehatan terpadu", "komprehensif",
            ],
        }
        lower = clean.lower()
        for label, kws in keywords.items():
            for kw in kws:
                if kw in lower:
                    return label

        return "Tidak Terklasifikasi"

    def _parse_reasoning_response(self, response: str) -> tuple:
        """Parse respons dengan format reasoning."""
        if not response:
            return "Tidak Terklasifikasi", ""

        alasan = ""
        tema_match = re.search(r"[Tt]ema:\s*(.+?)(?:\n|$)", response)
        if tema_match:
            tema = self._parse_response(tema_match.group(1).strip())
        else:
            tema = self._parse_response(response)

        alasan_match = re.search(r"[Aa]lasan:\s*(.+?)(?:\n[Tt]ema:|$)", response, re.DOTALL)
        if alasan_match:
            alasan = alasan_match.group(1).strip()

        return tema, alasan

    def classify_single(self, teks: str, with_reasoning: bool = False) -> dict:
        """Klasifikasi satu teks laporan."""
        if with_reasoning:
            prompt = get_zero_shot_prompt_with_reasoning(
                teks, self.tema_labels, self.tema_descriptions
            )
        else:
            prompt = get_zero_shot_prompt(
                teks, self.tema_labels, self.tema_descriptions
            )

        start = time.time()
        raw = self._call_ollama(prompt)
        elapsed = time.time() - start

        if with_reasoning:
            tema, alasan = self._parse_reasoning_response(raw)
        else:
            tema = self._parse_response(raw)
            alasan = ""

        return {
            "tema_prediksi": tema,
            "alasan": alasan,
            "raw_response": raw,
            "waktu_proses": round(elapsed, 2),
            "model": self.model,
        }

    def classify_batch(self, texts: list, with_reasoning: bool = False, delay: float = 0.5) -> list:
        """Klasifikasi batch teks."""
        results = []
        print(f"\n[INFO] Klasifikasi {len(texts)} teks | Model: {self.model}")
        print("-" * 50)

        for i, teks in enumerate(tqdm(texts, desc="Klasifikasi")):
            result = self.classify_single(teks, with_reasoning)
            result["index"] = i + 1
            results.append(result)
            if delay > 0 and i < len(texts) - 1:
                time.sleep(delay)

        classified = sum(1 for r in results if r["tema_prediksi"] != "Tidak Terklasifikasi")
        avg_time = sum(r["waktu_proses"] for r in results) / len(results)
        print(f"\n[INFO] Berhasil: {classified}/{len(texts)} | Rata-rata: {avg_time:.2f}s/teks")
        return results

    def classify_dataframe(self, df, text_column="teks_laporan",
                           with_reasoning=False, delay=0.5):
        """Klasifikasi dari DataFrame."""
        texts = df[text_column].tolist()
        results = self.classify_batch(texts, with_reasoning, delay)

        df_result = df.copy()
        df_result["tema_prediksi"] = [r["tema_prediksi"] for r in results]
        df_result["waktu_proses"] = [r["waktu_proses"] for r in results]
        df_result["raw_response"] = [r["raw_response"] for r in results]
        if with_reasoning:
            df_result["alasan"] = [r["alasan"] for r in results]

        return df_result

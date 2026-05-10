"""
Modul Klasifikasi Zero-Shot menggunakan LLM via Ollama.
Mengklasifikasikan teks laporan posyandu ke dalam tema
pelayanan kesehatan ibu dan anak tanpa pelatihan khusus.
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
    get_zero_shot_prompt_english,
)


class OllamaClassifier:
    """
    Klasifikasi Zero-Shot menggunakan Large Language Model via Ollama.
    Model mengklasifikasikan teks laporan posyandu ke dalam tema
    pelayanan KIA tanpa memerlukan data pelatihan berlabel.
    """

    def __init__(self, model: str = None, base_url: str = None):
        """
        Inisialisasi classifier.

        Args:
            model: Nama model Ollama (default dari config)
            base_url: URL endpoint Ollama (default dari config)
        """
        self.model = model or OLLAMA_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.api_url = f"{self.base_url}/api/generate"
        self.tema_labels = TEMA_LABELS
        self.tema_descriptions = TEMA_DESCRIPTIONS
        self.generation_config = GENERATION_CONFIG.copy()

    def check_connection(self) -> bool:
        """
        Cek koneksi ke Ollama server.

        Returns:
            True jika terhubung, False jika tidak
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                print(f"[INFO] Terhubung ke Ollama di {self.base_url}")
                print(f"[INFO] Model tersedia: {', '.join(model_names)}")

                # Cek apakah model yang dipilih tersedia
                model_available = any(
                    self.model in name for name in model_names
                )
                if not model_available:
                    print(f"[WARNING] Model '{self.model}' tidak ditemukan!")
                    print(f"[INFO] Jalankan: ollama pull {self.model}")
                    return False

                return True
            return False
        except requests.ConnectionError:
            print(f"[ERROR] Tidak dapat terhubung ke Ollama di {self.base_url}")
            print("[INFO] Pastikan Ollama sudah berjalan: ollama serve")
            return False
        except Exception as e:
            print(f"[ERROR] Kesalahan koneksi: {e}")
            return False

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Mengirim prompt ke Ollama API dan mendapatkan respons.

        Args:
            prompt: Prompt yang akan dikirim

        Returns:
            Respons teks dari LLM atau None jika gagal
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self.generation_config,
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"[ERROR] Ollama API error: {response.status_code}")
                print(f"        {response.text}")
                return None

        except requests.Timeout:
            print("[ERROR] Request timeout - model mungkin terlalu besar")
            return None
        except requests.ConnectionError:
            print("[ERROR] Koneksi ke Ollama terputus")
            return None
        except Exception as e:
            print(f"[ERROR] Kesalahan: {e}")
            return None

    def _parse_response(self, response: str) -> str:
        """
        Parse respons LLM untuk mengekstrak nama tema.

        Args:
            response: Raw response dari LLM

        Returns:
            Nama tema yang terdeteksi atau "Tidak Terklasifikasi"
        """
        if not response:
            return "Tidak Terklasifikasi"

        # Bersihkan response
        response_clean = response.strip().strip('"').strip("'").strip()

        # Cek exact match
        for label in self.tema_labels:
            if label.lower() == response_clean.lower():
                return label

        # Cek apakah response mengandung salah satu label
        for label in self.tema_labels:
            if label.lower() in response_clean.lower():
                return label

        # Cek partial match (kata kunci utama)
        label_keywords = {
            "Pemantauan Gizi Balita": [
                "pemantauan gizi", "gizi balita", "penimbangan",
                "status gizi", "stunting", "gizi kurang", "gizi baik",
                "berat badan", "tinggi badan", "antropometri",
            ],
            "Imunisasi Bayi": [
                "imunisasi", "vaksin", "vaksinasi", "bcg", "dpt",
                "polio", "campak", "kipi", "suntik",
            ],
            "Pemeriksaan Kesehatan Balita": [
                "pemeriksaan kesehatan", "kesehatan balita",
                "kesehatan terpadu", "pemeriksaan komprehensif",
                "pelayanan kesehatan", "kesehatan umum",
            ],
        }

        response_lower = response_clean.lower()
        for label, keywords in label_keywords.items():
            for keyword in keywords:
                if keyword in response_lower:
                    return label

        return "Tidak Terklasifikasi"

    def _parse_reasoning_response(self, response: str) -> tuple:
        """
        Parse respons yang mengandung reasoning (Chain-of-Thought).

        Args:
            response: Raw response dari LLM dengan format reasoning

        Returns:
            Tuple (tema, alasan)
        """
        if not response:
            return "Tidak Terklasifikasi", ""

        alasan = ""
        tema = ""

        # Cari pattern "Tema: xxx"
        tema_match = re.search(r"[Tt]ema:\s*(.+?)(?:\n|$)", response)
        if tema_match:
            tema_raw = tema_match.group(1).strip()
            tema = self._parse_response(tema_raw)

        # Cari pattern "Alasan: xxx"
        alasan_match = re.search(r"[Aa]lasan:\s*(.+?)(?:\n[Tt]ema:|$)", response, re.DOTALL)
        if alasan_match:
            alasan = alasan_match.group(1).strip()

        # Fallback: parse seluruh response jika tema masih kosong
        if tema == "Tidak Terklasifikasi" or not tema:
            tema = self._parse_response(response)

        return tema, alasan

    def classify_single(self, teks: str, with_reasoning: bool = False, use_english: bool = False) -> dict:
        """
        Klasifikasi satu teks laporan posyandu.

        Args:
            teks: Teks laporan yang akan diklasifikasikan
            with_reasoning: Jika True, minta LLM memberikan alasan
            use_english: Jika True, gunakan prompt bahasa Inggris

        Returns:
            Dictionary berisi hasil klasifikasi
        """
        # Buat prompt
        if with_reasoning:
            prompt = get_zero_shot_prompt_with_reasoning(
                teks, self.tema_labels, self.tema_descriptions
            )
        elif use_english:
            prompt = get_zero_shot_prompt_english(
                teks, self.tema_labels, self.tema_descriptions
            )
        else:
            prompt = get_zero_shot_prompt(
                teks, self.tema_labels, self.tema_descriptions
            )

        # Panggil LLM
        start_time = time.time()
        raw_response = self._call_ollama(prompt)
        elapsed_time = time.time() - start_time

        # Parse response
        if with_reasoning:
            tema, alasan = self._parse_reasoning_response(raw_response)
        else:
            tema = self._parse_response(raw_response)
            alasan = ""

        return {
            "teks": teks,
            "tema_prediksi": tema,
            "alasan": alasan,
            "raw_response": raw_response,
            "waktu_proses": round(elapsed_time, 2),
            "model": self.model,
        }

    def classify_batch(
        self,
        texts: list,
        with_reasoning: bool = False,
        use_english: bool = False,
        delay: float = 0.5
    ) -> list:
        """
        Klasifikasi batch teks laporan posyandu.

        Args:
            texts: List teks laporan yang akan diklasifikasikan
            with_reasoning: Jika True, minta LLM memberikan alasan
            use_english: Jika True, gunakan prompt bahasa Inggris
            delay: Jeda antar request (detik)

        Returns:
            List dictionary hasil klasifikasi
        """
        results = []

        print(f"\n[INFO] Memulai klasifikasi {len(texts)} teks...")
        print(f"[INFO] Model: {self.model}")
        print(f"[INFO] Mode: {'Reasoning' if with_reasoning else 'English' if use_english else 'Direct'}")
        print("-" * 60)

        for i, teks in enumerate(tqdm(texts, desc="Klasifikasi")):
            result = self.classify_single(teks, with_reasoning, use_english)
            result["index"] = i + 1
            results.append(result)

            # Jeda antar request
            if delay > 0 and i < len(texts) - 1:
                time.sleep(delay)

        # Statistik
        classified = sum(
            1 for r in results if r["tema_prediksi"] != "Tidak Terklasifikasi"
        )
        print("-" * 60)
        print(f"[INFO] Selesai! {classified}/{len(texts)} teks berhasil diklasifikasi")

        avg_time = sum(r["waktu_proses"] for r in results) / len(results)
        print(f"[INFO] Rata-rata waktu per teks: {avg_time:.2f} detik")

        return results

    def classify_dataframe(
        self,
        df,
        text_column: str = "teks_laporan",
        with_reasoning: bool = False,
        use_english: bool = False,
        delay: float = 0.5
    ):
        """
        Klasifikasi dari DataFrame pandas.

        Args:
            df: DataFrame dengan kolom teks laporan
            text_column: Nama kolom yang berisi teks
            with_reasoning: Jika True, minta LLM memberikan alasan
            use_english: Jika True, gunakan prompt bahasa Inggris
            delay: Jeda antar request

        Returns:
            DataFrame dengan kolom tambahan hasil klasifikasi
        """
        import pandas as pd

        texts = df[text_column].tolist()
        results = self.classify_batch(texts, with_reasoning, use_english, delay)

        # Tambahkan hasil ke DataFrame
        df_result = df.copy()
        df_result["tema_prediksi"] = [r["tema_prediksi"] for r in results]
        df_result["waktu_proses"] = [r["waktu_proses"] for r in results]
        df_result["raw_response"] = [r["raw_response"] for r in results]

        if with_reasoning:
            df_result["alasan"] = [r["alasan"] for r in results]

        return df_result

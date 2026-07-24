import os
import random
from typing import Callable, Dict, List, Optional

import pandas as pd

from config import (
    GIZI_CSV,
    KESEHATAN_CSV,
    IMUNISASI_CSV,
    GENERATED_DATA_PATH,
)


class TextGenerator:
    """
    Generator teks laporan Posyandu.

    Generator ini membuat teks untuk tiga tema:

    1. Gizi Balita
    2. Kesehatan Balita
    3. Imunisasi Bayi

    Setiap tema memiliki tiga tingkat kesulitan:

    - easy
    - medium
    - hard

    Distribusi tingkat kesulitan dibuat berbeda untuk setiap kelas.
    Kelas Imunisasi Bayi dibuat lebih jelas karena sebelumnya memiliki
    recall yang lebih rendah dibandingkan kelas lainnya.
    """

    LABEL_GIZI = "Gizi Balita"
    LABEL_KESEHATAN = "Kesehatan Balita"
    LABEL_IMUNISASI = "Imunisasi Bayi"

    VALID_DIFFICULTIES = [
        "easy",
        "medium",
        "hard",
    ]

    def __init__(
        self,
        seed: int = 42,
        easy_ratio: float = 0.50,
        medium_ratio: float = 0.40,
        hard_ratio: float = 0.10,
    ):
        """
        Args:
            seed:
                Random seed agar hasil dapat direproduksi.

            easy_ratio:
                Rasio default data mudah.

            medium_ratio:
                Rasio default data sedang.

            hard_ratio:
                Rasio default data sulit.
        """
        self.seed = seed
        self.random = random.Random(seed)

        self.easy_ratio = easy_ratio
        self.medium_ratio = medium_ratio
        self.hard_ratio = hard_ratio

        self.validate_ratios(
            easy_ratio=easy_ratio,
            medium_ratio=medium_ratio,
            hard_ratio=hard_ratio,
        )

        self.gizi_df = self.read_csv(
            path=GIZI_CSV,
            filename="gizi_balita.csv",
        )

        self.kesehatan_df = self.read_csv(
            path=KESEHATAN_CSV,
            filename="kesehatan_balita.csv",
        )

        self.imunisasi_df = self.read_csv(
            path=IMUNISASI_CSV,
            filename="imunisasi_bayi.csv",
        )

    # =====================================================
    # VALIDASI
    # =====================================================

    @staticmethod
    def validate_ratios(
        easy_ratio: float,
        medium_ratio: float,
        hard_ratio: float,
    ) -> None:
        """
        Memastikan seluruh rasio valid dan jumlahnya sama dengan 1.
        """
        ratios = {
            "easy_ratio": easy_ratio,
            "medium_ratio": medium_ratio,
            "hard_ratio": hard_ratio,
        }

        for name, value in ratios.items():
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} harus berada antara 0 dan 1. "
                    f"Nilai diterima: {value}"
                )

        total = (
            easy_ratio
            + medium_ratio
            + hard_ratio
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Jumlah easy_ratio, medium_ratio, dan hard_ratio "
                f"harus sama dengan 1.0. Jumlah saat ini: {total}"
            )

    @staticmethod
    def validate_difficulty_weights(
        difficulty_weights: Dict[str, float],
    ) -> None:
        """
        Memvalidasi distribusi tingkat kesulitan per kelas.
        """
        required_keys = {
            "easy",
            "medium",
            "hard",
        }

        missing_keys = (
            required_keys
            - set(difficulty_weights.keys())
        )

        if missing_keys:
            raise ValueError(
                "Difficulty weights tidak lengkap. "
                f"Key yang belum tersedia: {sorted(missing_keys)}"
            )

        for key in required_keys:
            value = difficulty_weights[key]

            if not 0 <= value <= 1:
                raise ValueError(
                    f"Nilai difficulty '{key}' harus antara 0 dan 1."
                )

        total = sum(
            difficulty_weights[key]
            for key in required_keys
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Jumlah difficulty weights harus sama dengan 1.0. "
                f"Jumlah saat ini: {total}"
            )

    # =====================================================
    # PEMBACAAN DATA
    # =====================================================

    @staticmethod
    def read_csv(
        path: str,
        filename: str,
    ) -> pd.DataFrame:
        """
        Membaca file CSV sumber.
        """
        if not os.path.exists(path):
            print(
                f"[WARNING] File tidak ditemukan: {filename}"
            )
            print(
                f"[WARNING] Lokasi yang diperiksa: {path}"
            )

            return pd.DataFrame()

        print(
            f"[INFO] Membaca data {filename}..."
        )

        try:
            df = pd.read_csv(path)

        except Exception as error:
            raise ValueError(
                f"Gagal membaca {filename}: {error}"
            ) from error

        print(
            f"[INFO] Jumlah baris {filename}: {len(df)}"
        )

        print(
            f"[INFO] Kolom {filename}:"
        )

        print(
            df.columns.tolist()
        )

        return df

    # =====================================================
    # UTILITAS
    # =====================================================

    @staticmethod
    def clean_value(
        value,
        default: str = "tidak tersedia",
    ) -> str:
        """
        Membersihkan nilai dari CSV.
        """
        if pd.isna(value):
            return default

        result = str(value).strip()

        if result == "":
            return default

        return result

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        """
        Menghapus baris dan spasi berlebih dari teks.
        """
        return " ".join(
            str(text).split()
        )

    def choose(
        self,
        options: List[str],
    ) -> str:
        """
        Memilih salah satu variasi teks secara acak.
        """
        if not options:
            raise ValueError(
                "Daftar pilihan teks tidak boleh kosong."
            )

        return self.random.choice(options)

    def determine_difficulty(
        self,
        difficulty_weights: Optional[
            Dict[str, float]
        ] = None,
    ) -> str:
        """
        Menentukan tingkat kesulitan.

        Apabila difficulty_weights tersedia, distribusi tersebut
        digunakan. Jika tidak, gunakan rasio default dari constructor.
        """
        if difficulty_weights is None:
            difficulty_weights = {
                "easy": self.easy_ratio,
                "medium": self.medium_ratio,
                "hard": self.hard_ratio,
            }

        self.validate_difficulty_weights(
            difficulty_weights
        )

        return self.random.choices(
            population=self.VALID_DIFFICULTIES,
            weights=[
                difficulty_weights["easy"],
                difficulty_weights["medium"],
                difficulty_weights["hard"],
            ],
            k=1,
        )[0]

    def sample_dataframe(
        self,
        df: pd.DataFrame,
        n: Optional[int],
    ) -> pd.DataFrame:
        """
        Mengambil sampel data dari DataFrame.

        Apabila jumlah data lebih sedikit dari n, sampling dilakukan
        dengan replacement.
        """
        if df.empty:
            return pd.DataFrame()

        if n is None:
            return df.copy()

        if n <= 0:
            raise ValueError(
                "n_per_tema harus lebih besar dari 0."
            )

        if len(df) >= n:
            return df.sample(
                n=n,
                replace=False,
                random_state=self.seed,
            ).copy()

        return df.sample(
            n=n,
            replace=True,
            random_state=self.seed,
        ).copy()

    # =====================================================
    # GENERATOR GIZI BALITA
    # =====================================================

    def generate_from_gizi(
        self,
        row: pd.Series,
        difficulty: Optional[str] = None,
    ) -> str:
        """
        Membuat teks untuk tema Gizi Balita.
        """
        difficulty = (
            difficulty
            or self.determine_difficulty()
        )

        nama = self.clean_value(
            row.get("Nama Balita")
        )

        identitas = self.clean_value(
            row.get("ID_Balita")
        )

        umur = self.clean_value(
            row.get("Umur (Bulan)")
        )

        jenis_kelamin = self.clean_value(
            row.get("Jenis Kelamin")
        )

        berat = self.clean_value(
            row.get("Berat Badan (kg)")
        )

        tinggi = self.clean_value(
            row.get("Tinggi Badan (cm)")
        )

        status_gizi = self.clean_value(
            row.get("Status Gizi")
        )

        keterangan = self.clean_value(
            row.get("Keterangan")
        )

        pembuka = self.choose([
            (
                "Balita datang mengikuti kegiatan rutin "
                "di posyandu."
            ),
            (
                "Petugas mencatat hasil kunjungan bulanan "
                "seorang balita."
            ),
            (
                "Orang tua membawa anak untuk melakukan "
                "pemantauan pertumbuhan."
            ),
            (
                "Kunjungan dilakukan untuk menilai perubahan "
                "pertumbuhan anak."
            ),
        ])

        if difficulty == "easy":
            text = f"""
            {pembuka}

            Balita bernama {nama}, dengan ID {identitas},
            berusia {umur} bulan dan berjenis kelamin
            {jenis_kelamin}.

            Petugas melakukan penimbangan berat badan dan
            pengukuran tinggi badan.

            Berat badan anak tercatat {berat} kg dan tinggi
            badan tercatat {tinggi} cm.

            Berdasarkan hasil pengukuran tersebut, status gizi
            anak adalah {status_gizi}.

            Catatan petugas menyebutkan {keterangan}.

            Orang tua diberikan edukasi mengenai pemenuhan
            gizi seimbang dan pemantauan pertumbuhan anak.
            """

        elif difficulty == "medium":
            text = f"""
            {pembuka}

            {nama}, usia {umur} bulan, menjalani pengukuran
            pertumbuhan secara rutin.

            Hasil penimbangan menunjukkan berat badan
            {berat} kg, sedangkan tinggi badan anak adalah
            {tinggi} cm.

            Kondisi gizi berdasarkan hasil pencatatan adalah
            {status_gizi}.

            Petugas memberikan catatan berupa {keterangan}
            dan meminta keluarga memperhatikan pola makan
            anak sampai kunjungan berikutnya.
            """

        else:
            text = self.choose([
                f"""
                Pada kunjungan bulan ini, {nama} menjalani
                penimbangan dan pengukuran tubuh.

                Hasilnya adalah {berat} kg dan {tinggi} cm.

                Berdasarkan hasil tersebut, kondisi pertumbuhan
                anak tercatat {status_gizi}.

                Keluarga diminta memperhatikan asupan harian
                dan datang kembali pada bulan berikutnya.
                """,

                f"""
                Petugas mencatat perubahan fisik {nama},
                usia {umur} bulan.

                Angka pada timbangan menunjukkan {berat} kg,
                sedangkan hasil pengukuran tinggi adalah
                {tinggi} cm.

                Catatan kondisi anak adalah {status_gizi},
                dengan keterangan tambahan {keterangan}.
                """,

                f"""
                Hasil pengukuran rutin {nama} menunjukkan
                berat badan {berat} kg dan tinggi badan
                {tinggi} cm.

                Petugas mencatat kondisi {status_gizi} serta
                memberikan arahan tentang pola makan dan
                pemantauan pertumbuhan.
                """,
            ])

        return self.normalize_text(text)

    # =====================================================
    # GENERATOR KESEHATAN BALITA
    # =====================================================

    def generate_from_kesehatan(
        self,
        row: pd.Series,
        difficulty: Optional[str] = None,
    ) -> str:
        """
        Membuat teks untuk tema Kesehatan Balita.
        """
        difficulty = (
            difficulty
            or self.determine_difficulty()
        )

        nama = self.clean_value(
            row.get("Nama Balita")
        )

        identitas = self.clean_value(
            row.get("ID_Balita")
        )

        umur = self.clean_value(
            row.get(
                "Umur (bulan)",
                row.get("Umur (Bulan)")
            )
        )

        jenis_kelamin = self.clean_value(
            row.get("Jenis Kelamin")
        )

        status_imunisasi = self.clean_value(
            row.get("Status Imunisasi")
        )

        status_kesehatan = self.clean_value(
            row.get("Status Kesehatan")
        )

        status_tumbuh_kembang = self.clean_value(
            row.get("Status Tumbuh Kembang")
        )

        riwayat_penyakit = self.clean_value(
            row.get("Riwayat Penyakit")
        )

        pembuka = self.choose([
            (
                "Balita mengikuti pemeriksaan rutin untuk "
                "menilai kondisi secara menyeluruh."
            ),
            (
                "Petugas melakukan pemeriksaan beberapa aspek "
                "kondisi anak dalam satu kunjungan."
            ),
            (
                "Kunjungan hari ini mencakup pemeriksaan kondisi "
                "umum dan perkembangan anak."
            ),
            (
                "Orang tua membawa balita untuk menjalani "
                "pemeriksaan kesehatan terpadu."
            ),
        ])

        if difficulty == "easy":
            text = f"""
            {pembuka}

            Balita bernama {nama}, dengan ID {identitas},
            berusia {umur} bulan dan berjenis kelamin
            {jenis_kelamin}.

            Kondisi kesehatan anak tercatat
            {status_kesehatan}.

            Status tumbuh kembang anak adalah
            {status_tumbuh_kembang}.

            Status imunisasi berdasarkan buku KIA tercatat
            {status_imunisasi}.

            Riwayat penyakit yang pernah dialami adalah
            {riwayat_penyakit}.

            Petugas memberikan saran berdasarkan keseluruhan
            hasil pemeriksaan kesehatan anak.
            """

        elif difficulty == "medium":
            text = f"""
            {pembuka}

            {nama}, usia {umur} bulan, menjalani pemeriksaan
            kondisi umum, perkembangan, riwayat penyakit,
            dan catatan pelayanan sebelumnya.

            Hasil pemeriksaan menunjukkan kondisi kesehatan
            {status_kesehatan}.

            Perkembangan anak tercatat
            {status_tumbuh_kembang}, sedangkan catatan
            imunisasi adalah {status_imunisasi}.

            Keluarga juga menyampaikan riwayat penyakit berupa
            {riwayat_penyakit}.

            Petugas memberikan arahan tindak lanjut sesuai
            seluruh hasil pemeriksaan.
            """

        else:
            text = self.choose([
                f"""
                Dalam satu kunjungan, petugas memeriksa kondisi
                umum dan perkembangan {nama}.

                Kondisi anak tercatat {status_kesehatan},
                sementara perkembangan anak adalah
                {status_tumbuh_kembang}.

                Buku KIA menunjukkan status imunisasi
                {status_imunisasi}.

                Riwayat penyakit yang pernah dicatat adalah
                {riwayat_penyakit}.
                """,

                f"""
                {nama}, usia {umur} bulan, datang untuk
                evaluasi rutin.

                Petugas meninjau kondisi fisik, perkembangan,
                riwayat penyakit, dan kelengkapan pelayanan
                pada buku KIA.

                Hasilnya menunjukkan kondisi
                {status_kesehatan} dan perkembangan
                {status_tumbuh_kembang}.
                """,

                f"""
                Pemeriksaan rutin {nama} mencakup beberapa
                aspek kondisi anak.

                Catatan menunjukkan kondisi kesehatan
                {status_kesehatan}, perkembangan
                {status_tumbuh_kembang}, serta status
                imunisasi {status_imunisasi}.

                Petugas memberikan rekomendasi berdasarkan
                seluruh temuan tersebut.
                """,
            ])

        return self.normalize_text(text)

    # =====================================================
    # GENERATOR IMUNISASI BAYI
    # =====================================================

    def generate_from_imunisasi(
        self,
        row: pd.Series,
        difficulty: Optional[str] = None,
    ) -> str:
        """
        Membuat teks untuk tema Imunisasi Bayi.

        Kelas ini dibuat lebih jelas karena pada evaluasi sebelumnya
        recall Imunisasi Bayi lebih rendah daripada kelas lain.
        """
        difficulty = (
            difficulty
            or self.determine_difficulty()
        )

        nama = self.clean_value(
            row.get("Nama Bayi")
        )

        identitas = self.clean_value(
            row.get("ID_Balita")
        )

        umur = self.clean_value(
            row.get("Umur (Bulan)")
        )

        jenis_kelamin = self.clean_value(
            row.get("Jenis Kelamin")
        )

        jenis_imunisasi = self.clean_value(
            row.get("Jenis Imunisasi")
        )

        status_imunisasi = self.clean_value(
            row.get("Status Imunisasi")
        )

        pembuka = self.choose([
            (
                "Bayi datang ke posyandu sesuai jadwal "
                "imunisasi berdasarkan usianya."
            ),
            (
                "Orang tua membawa bayi untuk melanjutkan "
                "jadwal pemberian vaksin."
            ),
            (
                "Petugas menerima kunjungan bayi berdasarkan "
                "jadwal imunisasi pada buku KIA."
            ),
            (
                "Bayi mengikuti pelayanan imunisasi rutin "
                "sesuai jadwal yang telah ditentukan."
            ),
        ])

        if difficulty == "easy":
            text = f"""
            {pembuka}

            Bayi bernama {nama}, dengan ID {identitas},
            berusia {umur} bulan dan berjenis kelamin
            {jenis_kelamin}.

            Pada kunjungan ini bayi menerima imunisasi
            {jenis_imunisasi}.

            Status kelengkapan imunisasi bayi tercatat
            {status_imunisasi}.

            Petugas menjelaskan manfaat vaksin, kemungkinan
            reaksi ringan setelah imunisasi, dan jadwal
            pemberian vaksin berikutnya.
            """

        elif difficulty == "medium":
            text = f"""
            {pembuka}

            {nama}, usia {umur} bulan, menerima vaksin
            {jenis_imunisasi} sesuai jadwal pada buku KIA.

            Catatan imunisasi anak saat ini adalah
            {status_imunisasi}.

            Setelah pemberian vaksin, petugas meminta orang tua
            mengamati kondisi bayi dan kembali sesuai jadwal
            imunisasi berikutnya.
            """

        else:
            text = self.choose([
                f"""
                {nama}, usia {umur} bulan, datang sesuai jadwal
                pemberian vaksin.

                Petugas memberikan {jenis_imunisasi} dan
                memperbarui catatan imunisasi pada buku KIA.

                Status imunisasi bayi tercatat
                {status_imunisasi}.
                """,

                f"""
                Pada kunjungan hari ini, {nama} menerima vaksin
                {jenis_imunisasi} sebagai bagian dari
                perlindungan terhadap penyakit.

                Petugas mencatat status imunisasi
                {status_imunisasi} dan menyampaikan jadwal
                vaksin berikutnya.
                """,

                f"""
                Buku KIA milik {nama} diperiksa untuk melihat
                kelengkapan imunisasi.

                Berdasarkan jadwal usia {umur} bulan,
                petugas memberikan vaksin
                {jenis_imunisasi}.

                Orang tua diminta mengamati reaksi bayi setelah
                imunisasi dan datang kembali sesuai jadwal.
                """,
            ])

        return self.normalize_text(text)

    # =====================================================
    # KONTEKS TAMBAHAN
    # =====================================================

    def create_ambiguous_variation(
        self,
        text: str,
        label: str,
    ) -> str:
        """
        Menambahkan konteks sekunder tanpa menghilangkan fokus utama.

        Konteks tambahan dibuat ringan agar dataset tetap realistis,
        tetapi tidak terlalu sulit seperti versi sebelumnya.
        """
        if label == self.LABEL_GIZI:
            additions = [
                (
                    "Petugas juga memeriksa buku KIA secara singkat, "
                    "tetapi fokus kunjungan tetap pada hasil "
                    "penimbangan dan pertumbuhan anak."
                ),
                (
                    "Kondisi umum anak diamati sebelum petugas "
                    "memberikan edukasi mengenai pola makan."
                ),
                (
                    "Orang tua juga diingatkan mengikuti jadwal "
                    "pelayanan kesehatan anak berikutnya."
                ),
            ]

        elif label == self.LABEL_KESEHATAN:
            additions = [
                (
                    "Petugas menilai seluruh informasi secara terpadu "
                    "sebelum memberikan rekomendasi."
                ),
                (
                    "Keluarga diminta memperhatikan pola makan, "
                    "perkembangan, dan jadwal kunjungan anak."
                ),
                (
                    "Catatan pertumbuhan dan imunisasi digunakan "
                    "sebagai bagian dari pemeriksaan menyeluruh."
                ),
            ]

        elif label == self.LABEL_IMUNISASI:
            additions = [
                (
                    "Kondisi umum bayi diperiksa terlebih dahulu "
                    "sebelum vaksin diberikan."
                ),
                (
                    "Petugas memastikan bayi dalam kondisi yang "
                    "memungkinkan untuk menerima imunisasi."
                ),
                (
                    "Berat badan bayi dicatat sebagai pemeriksaan "
                    "awal sebelum pemberian vaksin."
                ),
            ]

        else:
            additions = [
                (
                    "Petugas memberikan saran sesuai hasil "
                    "pelayanan hari ini."
                )
            ]

        additional_text = self.choose(
            additions
        )

        return self.normalize_text(
            f"{text} {additional_text}"
        )

    # =====================================================
    # PENAMBAHAN DATA PER KELAS
    # =====================================================

    def append_generated_rows(
        self,
        target: List[Dict],
        dataframe: pd.DataFrame,
        generator_function: Callable,
        label: str,
        n_per_tema: Optional[int],
        ambiguity_ratio: float,
        difficulty_weights: Dict[str, float],
    ) -> None:
        """
        Membuat teks untuk satu kelas dan menambahkannya ke dataset.
        """
        if not 0 <= ambiguity_ratio <= 1:
            raise ValueError(
                "ambiguity_ratio harus berada antara 0 dan 1."
            )

        self.validate_difficulty_weights(
            difficulty_weights
        )

        sampled_df = self.sample_dataframe(
            df=dataframe,
            n=n_per_tema,
        )

        if sampled_df.empty:
            print(
                f"[WARNING] Tidak ada data untuk tema {label}."
            )
            return

        for source_index, row in sampled_df.iterrows():
            difficulty = self.determine_difficulty(
                difficulty_weights=difficulty_weights
            )

            generated_text = generator_function(
                row=row,
                difficulty=difficulty,
            )

            is_ambiguous = (
                self.random.random()
                < ambiguity_ratio
            )

            if is_ambiguous:
                generated_text = (
                    self.create_ambiguous_variation(
                        text=generated_text,
                        label=label,
                    )
                )

            target.append({
                "teks_laporan": generated_text,
                "tema_aktual": label,
                "tingkat_kesulitan": difficulty,
                "ambigu": is_ambiguous,
                "sumber_index": source_index,
            })

    # =====================================================
    # GENERATE SELURUH DATA
    # =====================================================

    def generate_all(
        self,
        n_per_tema: Optional[int] = None,
        shuffle: bool = True,
        ambiguity_ratio: float = 0.10,
    ) -> pd.DataFrame:
        """
        Membuat seluruh dataset.

        Distribusi yang digunakan:

        Gizi Balita:
        - easy   55%
        - medium 35%
        - hard   10%
        - ambiguity sekitar 8%

        Kesehatan Balita:
        - easy   35%
        - medium 45%
        - hard   20%
        - ambiguity sekitar 12%

        Imunisasi Bayi:
        - easy   65%
        - medium 30%
        - hard    5%
        - ambiguity sekitar 5%

        Parameter ambiguity_ratio digunakan sebagai nilai dasar.
        Nilai aktual setiap kelas disesuaikan agar tidak terlalu sulit.
        """
        if not 0 <= ambiguity_ratio <= 1:
            raise ValueError(
                "ambiguity_ratio harus berada antara 0 dan 1."
            )

        generated_data: List[Dict] = []

        # -------------------------------------------------
        # GIZI BALITA
        # -------------------------------------------------

        if not self.gizi_df.empty:
            gizi_ambiguity = min(
                ambiguity_ratio,
                0.08,
            )

            self.append_generated_rows(
                target=generated_data,
                dataframe=self.gizi_df,
                generator_function=self.generate_from_gizi,
                label=self.LABEL_GIZI,
                n_per_tema=n_per_tema,
                ambiguity_ratio=gizi_ambiguity,
                difficulty_weights={
                    "easy": 0.55,
                    "medium": 0.35,
                    "hard": 0.10,
                },
            )

        # -------------------------------------------------
        # KESEHATAN BALITA
        # -------------------------------------------------

        if not self.kesehatan_df.empty:
            kesehatan_ambiguity = min(
                ambiguity_ratio,
                0.12,
            )

            self.append_generated_rows(
                target=generated_data,
                dataframe=self.kesehatan_df,
                generator_function=self.generate_from_kesehatan,
                label=self.LABEL_KESEHATAN,
                n_per_tema=n_per_tema,
                ambiguity_ratio=kesehatan_ambiguity,
                difficulty_weights={
                    "easy": 0.35,
                    "medium": 0.45,
                    "hard": 0.20,
                },
            )

        # -------------------------------------------------
        # IMUNISASI BAYI
        # -------------------------------------------------

        if not self.imunisasi_df.empty:
            imunisasi_ambiguity = min(
                ambiguity_ratio,
                0.05,
            )

            self.append_generated_rows(
                target=generated_data,
                dataframe=self.imunisasi_df,
                generator_function=self.generate_from_imunisasi,
                label=self.LABEL_IMUNISASI,
                n_per_tema=n_per_tema,
                ambiguity_ratio=imunisasi_ambiguity,
                difficulty_weights={
                    "easy": 0.65,
                    "medium": 0.30,
                    "hard": 0.05,
                },
            )

        result_df = pd.DataFrame(
            generated_data
        )

        if result_df.empty:
            raise ValueError(
                "Data hasil generator kosong. Pastikan file CSV "
                "gizi, kesehatan, dan imunisasi tersedia."
            )

        if shuffle:
            result_df = result_df.sample(
                frac=1,
                random_state=self.seed,
            ).reset_index(
                drop=True
            )

        return result_df

    # =====================================================
    # SIMPAN DATA
    # =====================================================

    def generate_and_save(
        self,
        n_per_tema: Optional[int] = None,
        shuffle: bool = True,
        ambiguity_ratio: float = 0.10,
    ) -> pd.DataFrame:
        """
        Membuat dataset kemudian menyimpannya ke CSV.
        """
        print(
            "\n[INFO] Membuat teks laporan..."
        )

        result_df = self.generate_all(
            n_per_tema=n_per_tema,
            shuffle=shuffle,
            ambiguity_ratio=ambiguity_ratio,
        )

        output_directory = os.path.dirname(
            GENERATED_DATA_PATH
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        result_df.to_csv(
            GENERATED_DATA_PATH,
            index=False,
        )

        print("\n" + "=" * 60)
        print("RINGKASAN DATA HASIL GENERATOR")
        print("=" * 60)

        print(
            f"Total teks laporan: {len(result_df)}"
        )

        print("\nDistribusi tema aktual:")

        print(
            result_df[
                "tema_aktual"
            ]
            .value_counts()
            .to_string()
        )

        print("\nDistribusi tingkat kesulitan:")

        print(
            result_df[
                "tingkat_kesulitan"
            ]
            .value_counts()
            .to_string()
        )

        total_ambiguous = int(
            result_df["ambigu"].sum()
        )

        ambiguous_percentage = (
            total_ambiguous
            / len(result_df)
            * 100
        )

        print(
            "\nJumlah data dengan konteks tambahan: "
            f"{total_ambiguous}/{len(result_df)} "
            f"({ambiguous_percentage:.2f}%)"
        )

        print(
            f"\n[INFO] File disimpan ke: "
            f"{GENERATED_DATA_PATH}"
        )

        return result_df

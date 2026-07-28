# ==========================================
# BAGIAN A: IMPORT LIBRARY
# ==========================================
import os
import streamlit as st
import requests
import json
import re
import uuid
import hashlib
import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime, date, timedelta

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AnyMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# BAGIAN B: KONFIGURASI OLLAMA LOKAL
# ==========================================
class OllamaPipelineWrapper:
    """Wrapper untuk memanggil Ollama API lokal (localhost:11434)."""
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def __call__(self, prompt, **kwargs):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1
        }
        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            text_result = response.json().get("response", "")
            return [{"generated_text": text_result}]
        except Exception as e:
            print(f"⚠️ Gagal menghubungi Ollama: {e}")
            return [{"generated_text": ""}]

@st.cache_resource
def load_llm_ollama(model_name: str = "llama3.2"):
    try:
        pipe = OllamaPipelineWrapper(model_name)
        return pipe
    except Exception:
        return None

llm_pipe = load_llm_ollama()

# ==========================================
# BAGIAN C: DATA PIPELINE (REVISI TOTAL)
# ==========================================

# --- Helper Path Asset ---
def get_asset_path(filename: str) -> str:
    """Mengembalikan path file di folder asset/ jika ada, atau fallback ke root."""
    asset_path = os.path.join("asset", filename)
    if os.path.exists(asset_path):
        return asset_path
    return filename


# --- C1: RAG dari SOP PDF ---
@st.cache_resource
def setup_rag_from_pdf(pdf_path: str = None):
    """Load SOP peminjaman ruang kelas ke ChromaDB untuk RAG."""
    if pdf_path is None:
        pdf_path = get_asset_path("sop_peminjaman.pdf")
    try:
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(raw_docs)
        vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db_sop")
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"⚠️ Gagal setup RAG ({pdf_path}): {e}")
        return None

# --- C2: Daftar Mahasiswa Aktif (FILE BARU: daftar_mahasiswa.xlsx) ---
@st.cache_data
def load_daftar_mahasiswa(path: str = None) -> pd.DataFrame:
    """
    Baca file master data mahasiswa dari BAAK.
    Sheet: 'DAFTAR MAHASISWA', header di baris ke-3 (header=2).
    Kolom: NO | NIM | NAMA | STATUS KEAKTIFAN | PROGRAM STUDI | ANGKATAN
    """
    if path is None:
        path = get_asset_path("daftar_mahasiswa.xlsx")
    try:
        df = pd.read_excel(path, sheet_name="DAFTAR MAHASISWA", header=2)
        df["NIM"] = df["NIM"].astype(str).str.strip()
        df["NAMA"] = df["NAMA"].astype(str).str.strip()
        df["STATUS KEAKTIFAN"] = df["STATUS KEAKTIFAN"].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"⚠️ Gagal load daftar mahasiswa ({path}): {e}")
        return pd.DataFrame(columns=["NO", "NIM", "NAMA", "STATUS KEAKTIFAN", "PROGRAM STUDI", "ANGKATAN"])

# --- C3: Riwayat Peminjaman Ruang (riwayat_peminjaman.xlsx) ---
@st.cache_data
def load_riwayat_peminjaman(path: str = None) -> pd.DataFrame:
    """
    Baca log peminjaman ruang kelas (booking insidental + status persetujuan).
    Sheet: 'PEMINJAMAN RUANG', header di baris ke-3 (header=2).
    Row 0 setelah header = sub-header [TGL, MULAI, SELESAI] -> di-skip.
    Kolom final: NO | TGL PENGAJUAN | HARI | TGL | MULAI | SELESAI | NAMA | NIM | RUANG |
                 KEPERLUAN / KEGIATAN / ACARA | KETERANGAN
    KETERANGAN: 'Disetujui', 'Menunggu konfirmasi', 'Dibatalkan'
    """
    if path is None:
        path = get_asset_path("riwayat_peminjaman.xlsx")
    try:
        df = pd.read_excel(path, sheet_name="PEMINJAMAN RUANG", header=2)
        df.columns = [str(c).strip() for c in df.columns]
        # Rename kolom merged header Excel:
        # 'KEGIATAN / TGL / JAM' → 'TGL', 'Unnamed: 4' → 'MULAI', 'Unnamed: 5' → 'SELESAI'
        rename_map = {
            "KEGIATAN / TGL / JAM": "TGL",
            "Unnamed: 4": "MULAI",
            "Unnamed: 5": "SELESAI",
        }
        df = df.rename(columns=rename_map)
        # Baris pertama adalah sub-header (TGL/MULAI/SELESAI) — skip
        df = df.iloc[1:].reset_index(drop=True)
        df = df.dropna(subset=["NIM"])
        # Parse tanggal
        df["TGL PENGAJUAN"] = pd.to_datetime(df["TGL PENGAJUAN"], format="%d/%m/%Y", errors="coerce")
        df["TGL"] = pd.to_datetime(df["TGL"], format="%d/%m/%Y", errors="coerce")
        # Bersihkan kolom teks
        df["RUANG"] = df["RUANG"].astype(str).str.strip()
        df["NIM"] = df["NIM"].astype(str).str.strip()
        df["KETERANGAN"] = df["KETERANGAN"].astype(str).str.strip()
        df["MULAI"] = df["MULAI"].astype(str).str.strip()
        df["SELESAI"] = df["SELESAI"].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"⚠️ Gagal load riwayat peminjaman ({path}): {e}")
        return pd.DataFrame(columns=["NO", "TGL PENGAJUAN", "HARI", "TGL", "MULAI", "SELESAI",
                                      "NAMA", "NIM", "RUANG", "KEPERLUAN / KEGIATAN / ACARA", "KETERANGAN"])

# --- C4: Jadwal Kuliah Reguler (jadwal_praktikum.xlsx) ---
def clean_room_code(value) -> str:
    """
    Perbaiki kode ruangan yang salah ter-konversi Excel jadi tanggal.
    Contoh: kode '3.5.3' terbaca sebagai Timestamp('2003-05-03').
    Konversi balik: tahun%100 . bulan . hari -> 3.5.3
    """
    if isinstance(value, pd.Timestamp) or "Timestamp" in str(type(value)):
        try:
            yy = value.year % 100
            return f"{yy}.{value.month}.{value.day}"
        except Exception:
            return str(value).strip()
    if isinstance(value, datetime):
        yy = value.year % 100
        return f"{yy}.{value.month}.{value.day}"
    return str(value).strip()

@st.cache_data
def load_jadwal_kuliah(path: str = None) -> pd.DataFrame:
    """
    Baca jadwal kuliah reguler mingguan (bentrok berulang per hari, bukan per tanggal).
    Sheet: 'Table 1', header di baris ke-2 (row 0=judul merged, row 1=header kolom).
    Kolom: NO | HARI | DOSEN | (separator) | JAM | RUANG | MATA KULIAH
    """
    if path is None:
        path = get_asset_path("jadwal_praktikum.xlsx")
    try:
        df = pd.read_excel(path, sheet_name="Table 1", header=0)
        # Row 0 setelah read = header asli [NO, HARI, DOSEN, _, JAM, RUANG, MATA KULIAH]
        # Skip row 0 (yang berisi header text) dan mulai dari data (row 1+)
        df.columns = ["NO", "HARI", "DOSEN", "_SEP", "JAM", "RUANG", "MATA_KULIAH"]
        df = df.iloc[1:].reset_index(drop=True)  # skip row header text
        df = df.drop(columns=["_SEP"]).dropna(subset=["HARI"])
        # Fix kode ruangan yang ter-konversi Excel jadi datetime
        df["RUANG"] = df["RUANG"].apply(clean_room_code)
        df["HARI"] = df["HARI"].astype(str).str.strip().str.upper()
        df["JAM"] = df["JAM"].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"⚠️ Gagal load jadwal kuliah ({path}): {e}")
        return pd.DataFrame(columns=["NO", "HARI", "DOSEN", "JAM", "RUANG", "MATA_KULIAH"])

# --- Inisialisasi Data Global ---
retriever = setup_rag_from_pdf()
mahasiswa_df = load_daftar_mahasiswa()
riwayat_df = load_riwayat_peminjaman()
jadwal_kuliah_df = load_jadwal_kuliah()

# Daftar semua kode ruangan unik (untuk saran alternatif)
ALL_ROOMS = sorted(riwayat_df["RUANG"].dropna().unique().tolist()) if not riwayat_df.empty else []

# ==========================================
# BAGIAN D: TOOLS & FUNGSI BISNIS
# ==========================================

NIM_PATTERN = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")

def verify_student_valid(nim: str) -> Dict[str, Any]:
    """
    Validasi mahasiswa ke file master daftar_mahasiswa.xlsx.
    Langkah 1: Cek format NIM (XX.XX.XXXX).
    Langkah 2: Cek apakah NIM terdaftar dan STATUS KEAKTIFAN = 'Aktif'.
    """
    nim = nim.strip()
    if not NIM_PATTERN.match(nim):
        return {"valid": False, "reason": f"Format NIM '{nim}' tidak sesuai standar AMIKOM (XX.XX.XXXX)."}

    if mahasiswa_df.empty:
        # Fallback: kalau file master tidak tersedia, validasi format saja
        return {"valid": True, "reason": f"Format NIM {nim} valid (file master tidak tersedia, validasi format saja)."}

    match = mahasiswa_df[mahasiswa_df["NIM"] == nim]
    if match.empty:
        return {"valid": False, "reason": f"NIM {nim} tidak terdaftar di database mahasiswa AMIKOM."}

    status = match.iloc[0]["STATUS KEAKTIFAN"]
    nama = match.iloc[0]["NAMA"]
    if status == "Aktif":
        return {"valid": True, "reason": f"Mahasiswa {nama} (NIM {nim}) — status: Aktif ✅"}
    else:
        return {"valid": False, "reason": f"Mahasiswa {nama} (NIM {nim}) — status: {status} ❌ (harus Aktif untuk meminjam)."}


def check_h3(tgl_kegiatan_str: str, min_days: int = 3) -> Dict[str, Any]:
    """
    Cek aturan H-3: tanggal kegiatan harus minimal 3 hari kalender dari hari ini.
    Menggunakan hari kalender biasa (bukan hari kerja).
    """
    try:
        tgl_kegiatan = datetime.strptime(tgl_kegiatan_str, "%Y-%m-%d").date()
        delta = (tgl_kegiatan - date.today()).days
        if delta >= min_days:
            return {"ok": True, "days_left": delta, "reason": f"Pengajuan H-{delta} hari, memenuhi syarat minimal H-{min_days}. ✅"}
        return {"ok": False, "days_left": delta, "reason": f"Pengajuan H-{delta} hari, melanggar minimal H-{min_days}. ❌"}
    except ValueError:
        return {"ok": False, "days_left": None, "reason": f"Format tanggal '{tgl_kegiatan_str}' tidak valid (harus YYYY-MM-DD)."}


def parse_time_range(time_str: str):
    """Parse time range string '07:00 - 08:40' menjadi (start_minutes, end_minutes)."""
    try:
        parts = time_str.replace(" ", "").split("-")
        if len(parts) == 2:
            start = parts[0].split(":")
            end = parts[1].split(":")
            start_min = int(start[0]) * 60 + int(start[1])
            end_min = int(end[0]) * 60 + int(end[1])
            return start_min, end_min
    except Exception:
        pass
    return None, None


def time_ranges_overlap(range1_str: str, range2_str: str) -> bool:
    """Cek apakah dua rentang waktu beririsan."""
    s1, e1 = parse_time_range(range1_str)
    s2, e2 = parse_time_range(range2_str)
    if s1 is None or s2 is None:
        # Kalau gagal parse, anggap bentrok (lebih aman)
        return True
    return s1 < e2 and s2 < e1


def check_bentrok(ruang: str, tgl: date, mulai: str, selesai: str) -> Dict[str, Any]:
    """
    Cek bentrok ganda:
    1. Ke booking insidental (riwayat_peminjaman.xlsx) yang KETERANGAN = 'Disetujui'
    2. Ke jadwal kuliah reguler mingguan (jadwal_praktikum.xlsx)
    Juga laporkan soft-conflict (booking 'Menunggu konfirmasi').
    """
    request_time_range = f"{mulai} - {selesai}"
    hari_map = ["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU", "MINGGU"]
    hari_str = hari_map[tgl.weekday()]
    warning = ""

    # 1) Cek bentrok ke booking insidental yang SUDAH disetujui
    if not riwayat_df.empty:
        approved = riwayat_df[
            (riwayat_df["RUANG"] == ruang) &
            (riwayat_df["TGL"].dt.date == tgl) &
            (riwayat_df["KETERANGAN"] == "Disetujui")
        ]
        for _, row in approved.iterrows():
            existing_range = f"{row['MULAI']} - {row['SELESAI']}"
            if time_ranges_overlap(request_time_range, existing_range):
                return {
                    "available": False,
                    "reason": f"Ruang {ruang} sudah dipakai (booking disetujui) pada {tgl} jam {existing_range}.",
                    "warning": ""
                }

        # Soft-conflict: booking 'Menunggu konfirmasi'
        pending = riwayat_df[
            (riwayat_df["RUANG"] == ruang) &
            (riwayat_df["TGL"].dt.date == tgl) &
            (riwayat_df["KETERANGAN"] == "Menunggu konfirmasi")
        ]
        for _, row in pending.iterrows():
            existing_range = f"{row['MULAI']} - {row['SELESAI']}"
            if time_ranges_overlap(request_time_range, existing_range):
                warning = f"⚠️ Ada pengajuan lain yang masih menunggu konfirmasi di ruang {ruang} pada {tgl} jam {existing_range}."
                break

    # 2) Cek bentrok ke jadwal kuliah reguler mingguan
    if not jadwal_kuliah_df.empty:
        reguler = jadwal_kuliah_df[
            (jadwal_kuliah_df["RUANG"] == ruang) &
            (jadwal_kuliah_df["HARI"] == hari_str)
        ]
        for _, row in reguler.iterrows():
            jam_kuliah = str(row["JAM"]).strip()
            if time_ranges_overlap(request_time_range, jam_kuliah):
                mk = row.get("MATA_KULIAH", "")
                return {
                    "available": False,
                    "reason": f"Ruang {ruang} terpakai jadwal kuliah reguler ({hari_str}, {jam_kuliah}: {mk}).",
                    "warning": ""
                }

    return {"available": True, "reason": "Tidak ada bentrok. ✅", "warning": warning}


def decide_approval(nim: str, ruang: str, tgl_kegiatan_str: str, mulai: str, selesai: str) -> Dict[str, Any]:
    """
    Implementasi eksak rumus approval tunggal:
    DISETUJUI ⟺ mahasiswa valid AND jadwal tidak bentrok AND H-3 terpenuhi
    """
    student = verify_student_valid(nim)
    h3 = check_h3(tgl_kegiatan_str)

    try:
        tgl_kegiatan = datetime.strptime(tgl_kegiatan_str, "%Y-%m-%d").date()
    except ValueError:
        return {
            "approved": False,
            "student_check": student,
            "h3_check": h3,
            "bentrok_check": {"available": False, "reason": "Tanggal tidak valid.", "warning": ""},
            "message": f"❌ Format tanggal kegiatan tidak valid: {tgl_kegiatan_str}"
        }

    bentrok = check_bentrok(ruang, tgl_kegiatan, mulai, selesai)
    approved = student["valid"] and bentrok["available"] and h3["ok"]

    return {
        "approved": approved,
        "student_check": student,
        "h3_check": h3,
        "bentrok_check": bentrok,
    }


def find_alternative_room(exclude_room: str, tgl: date, mulai: str, selesai: str) -> Optional[str]:
    """
    Cari ruangan alternatif dari 99 kode ruangan unik yang tersedia,
    di hari/jam yang sama, yang lolos check_bentrok.
    """
    for ruang in ALL_ROOMS:
        if ruang == exclude_room:
            continue
        hasil = check_bentrok(ruang, tgl, mulai, selesai)
        if hasil["available"]:
            return ruang
    return None


def allocate_inventory(items_needed: List[str]) -> Dict[str, Any]:
    """Cek ketersediaan inventaris (proyektor, mic, dll)."""
    available_items = ["microphone", "proyektor", "mic", "spidol", "penghapus"]
    status = {item: (item.lower() in available_items) for item in items_needed}
    all_available = all(status.values()) if items_needed else True
    work_order = {
        "work_order_id": f"WO-{uuid.uuid4().hex[:8].upper()}",
        "items_requested": items_needed,
        "status_ketersediaan": status,
    }
    return {"all_available": all_available, "status": status, "work_order": work_order}


# --- Helper: Tanggal Relatif & Multi-Turn ---

def parse_indonesian_date_text(text: str) -> Optional[str]:
    """
    Parse tanggal Indonesia format teks seperti '2 agustus', '15 agustus 2026', '5 juli', dll.
    Mengembalikan format YYYY-MM-DD.
    """
    text_lower = text.lower().strip()
    month_map = {
        "januari": 1, "jan": 1,
        "februari": 2, "feb": 2,
        "maret": 3, "mar": 3,
        "april": 4, "apr": 4,
        "mei": 5,
        "juni": 6, "jun": 6,
        "juli": 7, "jul": 7,
        "agustus": 8, "agust": 8, "agt": 8,
        "september": 9, "sep": 9, "sept": 9,
        "oktober": 10, "okt": 10,
        "november": 11, "nov": 11,
        "desember": 12, "des": 12
    }
    pattern = r"\b(\d{1,2})\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|jan|feb|mar|apr|jun|jul|agust|agt|sep|sept|okt|nov|des)\s*(\d{4})?\b"
    match = re.search(pattern, text_lower)
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        year_str = match.group(3)

        month = month_map.get(month_str)
        year = int(year_str) if year_str else date.today().year

        if month and 1 <= day <= 31:
            try:
                dt = date(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    return None


def resolve_relative_date(text: str):
    """
    Parse tanggal relatif dari teks bahasa Indonesia.
    Mengembalikan format YYYY-MM-DD atau None jika tidak ditemukan.
    Contoh: 'besok' -> date.today()+1, '2 agustus' -> 2026-08-02
    """
    text_lower = text.lower().strip()
    today = date.today()

    # Cek tanggal teks Indonesia (misal: "2 agustus")
    indo_dt = parse_indonesian_date_text(text_lower)
    if indo_dt:
        return indo_dt

    if "hari ini" in text_lower:
        return today.strftime("%Y-%m-%d")
    if "besok" in text_lower or "bsk" in text_lower:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if "lusa" in text_lower:
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")
    if "minggu depan" in text_lower and "hari minggu" not in text_lower:
        return (today + timedelta(days=7)).strftime("%Y-%m-%d")

    # Hari spesifik: "senin depan", "hari selasa", "kamis", dll.
    hari_map = {
        "senin": 0, "selasa": 1, "rabu": 2, "kamis": 3,
        "jumat": 4, "jum'at": 4, "sabtu": 5, "minggu": 6
    }
    for hari_name, target_weekday in hari_map.items():
        if re.search(rf"\b{hari_name}\b", text_lower):
            current_weekday = today.weekday()
            days_ahead = target_weekday - current_weekday
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    return None


def extract_entities_lightweight(text: str) -> dict:
    """
    Ekstraksi entitas ringan via regex saja (tanpa LLM).
    Digunakan untuk merge pada multi-turn conversation.
    """
    result = {}
    text_lower = text.lower().strip()

    # NIM
    nim_match = re.search(r"\d{2}\.\d{2}\.\d{4}", text)
    if nim_match:
        result["nim"] = nim_match.group()

    # Room ID (cari semua match bertitik dan pilih yang bukan NIM)
    all_room_matches = re.findall(r"(?:L\s?)?\d+\.\d+\.\d+", text)
    for candidate in all_room_matches:
        cand_clean = candidate.strip()
        if cand_clean != result.get("nim"):
            result["room_id"] = cand_clean
            break

    # Date (YYYY-MM-DD atau relatif)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if date_match:
        result["date"] = date_match.group()
    else:
        rel = resolve_relative_date(text)
        if rel:
            result["date"] = rel

    # Time range / single time
    time_match = re.search(r"(\d{1,2})[:.](\d{2})\s*[-\u2013]\s*(\d{1,2})[:.](\d{2})", text)
    if time_match:
        result["time_start"] = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
        result["time_end"] = f"{int(time_match.group(3)):02d}:{time_match.group(4)}"
    else:
        single = re.search(r"(?:jam|pukul)?\s*(\d{1,2})[:.](\d{2})", text_lower)
        if single:
            result["time_start"] = f"{int(single.group(1)):02d}:{single.group(2)}"

    # Nama
    nama_match = re.search(r"(?:saya|nama)\s+([A-Za-z\s]{3,30}?)(?=\s*,|\s+NIM|\s+nim|\s+ingin|\s+mau|$)", text, re.IGNORECASE)
    if nama_match:
        result["nama"] = nama_match.group(1).strip()

    return result


def build_synthetic_prompt(entities: dict) -> str:
    """
    Bangun kalimat sintetis dari entities yang terkumpul
    untuk dikirim ulang ke graph sebagai prompt lengkap.
    """
    parts = []
    if entities.get("nama"):
        parts.append(f"Saya {entities['nama']}")
    if entities.get("nim"):
        parts.append(f"NIM {entities['nim']}")
    if entities.get("room_id"):
        parts.append(f"ingin pinjam ruang {entities['room_id']}")
    if entities.get("date"):
        parts.append(f"tanggal {entities['date']}")
    if entities.get("time_start"):
        time_str = entities["time_start"]
        if entities.get("time_end"):
            time_str += f"-{entities['time_end']}"
        parts.append(f"jam {time_str}")
    if entities.get("keperluan"):
        parts.append(f"untuk {entities['keperluan']}")
    return ", ".join(parts) if parts else ""


# ==========================================
# BAGIAN E: STATE, PROMPTS, NODES
# ==========================================

class AgentState(TypedDict):
    student_id: str          # NIM mahasiswa
    student_name: str        # Nama mahasiswa (dari input)
    room_id: str             # Kode ruangan yang diminta
    date: str                # Tanggal kegiatan (YYYY-MM-DD)
    time_start: str          # Jam mulai (HH:MM)
    time_end: str            # Jam selesai (HH:MM)
    keperluan: str           # Keperluan/kegiatan/acara
    items_needed: List[str]  # Item inventaris yang diminta
    retrieved_context: str   # Konteks dari RAG (SOP PDF)
    data_complete: Optional[bool]  # True jika semua data wajib lengkap
    # Hasil pengecekan
    student_check: Optional[Dict]
    h3_check: Optional[Dict]
    bentrok_check: Optional[Dict]
    approval_result: Optional[bool]
    # Output
    digital_pass: Optional[str]
    suggested_alternative: Optional[str]
    work_order: Optional[Dict[str, Any]]
    decision_log: List[str]
    message: str
    messages: Annotated[List[AnyMessage], add_messages]


# --- Prompt Templates ---

ENTITY_EXTRACTION_PROMPT = """Kamu adalah asisten yang mengekstrak entitas dari permintaan peminjaman ruangan kampus.
Ekstrak entitas berikut dari kalimat mahasiswa, dalam format JSON:
{{"nim": str atau null, "nama": str atau null, "room_id": str atau null,
  "date": "YYYY-MM-DD" atau null, "time_start": "HH:MM" atau null,
  "time_end": "HH:MM" atau null, "keperluan": str atau null,
  "items_needed": [str]}}

Catatan:
- NIM format AMIKOM: XX.XX.XXXX (contoh: 25.11.1605)
- Kode ruangan contoh: 5.1.4, 2.3.1, 1.1.1, L 2.4.4
- Kalau waktu disebutkan sebagai range "09:00-12:00", pisahkan jadi time_start dan time_end
- Hari ini adalah tanggal: {today_str}. Jika tanggal disebutkan relatif ("besok", "lusa", "senin depan"), konversikan ke format YYYY-MM-DD berdasarkan hari ini.
- Kalau tidak disebutkan, isi null / list kosong
- HANYA keluarkan JSON, tanpa teks tambahan

Contoh 1:
Kalimat: "Saya Eko Saputra NIM 19.36.1640, ingin pinjam ruang 5.1.4 tanggal 2 Januari 2026 jam 09:00-12:00 untuk praktikum basis data, butuh proyektor"
JSON: {{"nim": "19.36.1640", "nama": "Eko Saputra", "room_id": "5.1.4", "date": "2026-01-02", "time_start": "09:00", "time_end": "12:00", "keperluan": "praktikum basis data", "items_needed": ["proyektor"]}}

Contoh 2:
Kalimat: "Mau pinjam ruang 2.1.6 buat seminar proposal hari Selasa depan"
JSON: {{"nim": null, "nama": null, "room_id": "2.1.6", "date": null, "time_start": null, "time_end": null, "keperluan": "seminar proposal", "items_needed": []}}

Contoh 3:
Kalimat: "Nama saya Mahendra Fauzi, NIM 25.11.1605, mau booking ruang 1.1.1 tanggal 15 Agustus 2026 jam 13:00-15:00 untuk rapat UKM, perlu mic dan proyektor"
JSON: {{"nim": "25.11.1605", "nama": "Mahendra Fauzi", "room_id": "1.1.1", "date": "2026-08-15", "time_start": "13:00", "time_end": "15:00", "keperluan": "rapat UKM", "items_needed": ["mic", "proyektor"]}}

Sekarang ekstrak dari kalimat berikut:
Kalimat: "{user_prompt}"
JSON:"""

COMPOSE_PROMPT = """Anda adalah asisten AI peminjaman ruangan kampus Universitas AMIKOM Yogyakarta.
Tugas Anda: Menyampaikan keputusan peminjaman ruangan kepada mahasiswa secara singkat, ramah, dan langsung pada inti masalah (maksimal 3 kalimat).

ATURAN STRICT:
1. DILARANG mengarang link/URL apapun (seperti bit.ly, http, dll).
2. DILARANG membuat surat formal panjang dengan salam penutup "[nama asisten]" atau mengulang-ulang data lama yang sudah ditolak.
3. Fokus HANYA pada hasil keputusan akhir berikut:
   {decision}

4. Sampaikan keputusan dengan bahasa Indonesia yang ramah, sopan, dan santai.

Susun balasan singkat untuk mahasiswa:"""


# --- Node Functions ---

def node_input(state: AgentState) -> AgentState:
    """
    Ekstraksi entitas via LLM (JSON terstruktur) dengan fallback regex.
    TIDAK berasumsi data yang tidak diberikan user — jika data wajib tidak lengkap,
    set data_complete=False dan minta user melengkapi.
    """
    user_prompt = state["messages"][-1].content if state.get("messages") else ""
    extracted = {}
    log = state.get("decision_log", [])

    # Coba ekstraksi via LLM
    if llm_pipe is not None:
        try:
            raw = llm_pipe(ENTITY_EXTRACTION_PROMPT.format(user_prompt=user_prompt, today_str=date.today().strftime('%Y-%m-%d')))[0]["generated_text"]
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_part = raw[json_start:json_end]
                extracted = json.loads(json_part)
                log.append("[Input Node] ✅ Ekstraksi entitas via LLM berhasil.")
        except Exception as e:
            log.append(f"[Input Node] ⚠️ LLM extraction gagal ({e}), pakai fallback regex.")

    user_lower = user_prompt.lower()

    # --- Ekstraksi tanpa asumsi (TIDAK ada default value) ---

    # NIM: pattern XX.XX.XXXX
    nim = extracted.get("nim") or None
    if not nim:
        nim_match = re.search(r"\d{2}\.\d{2}\.\d{4}", user_prompt)
        nim = nim_match.group() if nim_match else None

    # Nama
    nama = extracted.get("nama") or None

    # Room ID
    room_id = extracted.get("room_id") or None
    if not room_id:
        room_matches = re.findall(r"(?:L\s?)?\d+\.\d+\.\d+", user_prompt)
        for candidate in room_matches:
            cand_clean = candidate.strip()
            if cand_clean != nim:
                room_id = cand_clean
                break

    # Tanggal: coba dari LLM → fallback regex YYYY-MM-DD → fallback tanggal relatif
    date_str = extracted.get("date") or None
    if not date_str:
        date_regex = re.search(r"\d{4}-\d{2}-\d{2}", user_prompt)
        if date_regex:
            date_str = date_regex.group()
            log.append(f"[Input Node] \U0001f4c5 Tanggal ditemukan via regex \u2192 {date_str}")
    if not date_str:
        date_str = resolve_relative_date(user_prompt)
        if date_str:
            log.append(f"[Input Node] \U0001f4c5 Tanggal relatif terdeteksi \u2192 {date_str}")

    # Waktu: TIDAK ada default, harus dari user
    time_start = extracted.get("time_start") or None
    time_end = extracted.get("time_end") or None
    # Fallback regex untuk waktu dari teks
    if not time_start:
        time_match = re.search(r"(\d{1,2})[:.](\d{2})\s*[-–]\s*(\d{1,2})[:.](\d{2})", user_prompt)
        if time_match:
            time_start = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
            time_end = f"{int(time_match.group(3)):02d}:{time_match.group(4)}"
        else:
            # Coba deteksi jam tunggal (tanpa range)
            single_time = re.search(r"jam\s*(\d{1,2})[:.](\d{2})", user_lower)
            if single_time:
                time_start = f"{int(single_time.group(1)):02d}:{single_time.group(2)}"
                # time_end tetap None — user harus melengkapi

    # Keperluan
    keperluan = extracted.get("keperluan") or None

    # Items needed
    items_needed = extracted.get("items_needed") or []
    if not items_needed:
        if "proyektor" in user_lower:
            items_needed.append("proyektor")
        if "mic" in user_lower or "microphone" in user_lower:
            items_needed.append("mic")

    # --- Cek kelengkapan data wajib ---
    REQUIRED_FIELDS = {
        "NIM": nim,
        "Kode ruangan": room_id,
        "Tanggal kegiatan (format: YYYY-MM-DD, contoh: 2026-08-15)": date_str,
        "Jam mulai": time_start,
    }
    missing = [name for name, val in REQUIRED_FIELDS.items() if not val]

    if missing:
        # Data tidak lengkap — susun balasan ramah & casual (bukan error template)
        log.append(f"[Input Node] ⚠️ Data kurang lengkap: {', '.join(missing)}")

        missing_labels = []
        if not nim:
            missing_labels.append("NIM")
        if not room_id:
            missing_labels.append("Kode ruangan")
        if not date_str:
            missing_labels.append("Tanggal kegiatan")
        if not time_start:
            missing_labels.append("Jam mulai")

        missing_text = " dan ".join(missing_labels) if len(missing_labels) <= 2 else ", ".join(missing_labels[:-1]) + f", serta {missing_labels[-1]}"

        # Bangun greeting yang ramah
        user_display = nama or (f"NIM {nim}" if nim else "Mahasiswa")

        if room_id and nim and (not date_str or not time_start):
            casual_msg = (f"Hai **{user_display}**! 👋 Peminjaman untuk **Ruang {room_id}** sudah saya catat.\n\n"
                          f"Boleh infokan **{missing_text}** agar bisa langsung saya proseskan ketersediaannya?")
        elif date_str and time_start and (not room_id or not nim):
            casual_msg = (f"Halo **{user_display}**! 👋 Jadwal tanggal **{date_str}** jam **{time_start}** sudah dicatat.\n\n"
                          f"Boleh infokan **{missing_text}** untuk melanjutkan peminjaman ruangan?")
        else:
            casual_msg = (f"Halo **{user_display}**! 👋 Untuk membantu proses peminjaman ruangan, "
                          f"saya masih memerlukan **{missing_text}**.\n\n"
                          f"Silakan infokan di bawah ini ya! 😊")

        return {
            "student_id": nim or "",
            "student_name": nama or "",
            "room_id": room_id or "",
            "date": date_str or "",
            "time_start": time_start or "",
            "time_end": time_end or "",
            "keperluan": keperluan or "",
            "items_needed": items_needed,
            "data_complete": False,
            "message": casual_msg,
            "decision_log": log,
        }

    # Data lengkap — lanjut ke validasi
    log.append(f"[Input Node] ✅ Data lengkap: NIM={nim}, Ruang={room_id}, "
               f"Tanggal={date_str}, Jam={time_start}-{time_end or '?'}")

    return {
        "student_id": nim,
        "student_name": nama or "",
        "room_id": room_id,
        "date": date_str,
        "time_start": time_start,
        "time_end": time_end or time_start,  # kalau end tidak ada, samakan dengan start
        "keperluan": keperluan or "",
        "items_needed": items_needed,
        "data_complete": True,
        "decision_log": log,
    }


def node_rag_retrieval(state: AgentState) -> AgentState:
    """Ambil konteks SOP relevan dari ChromaDB via RAG."""
    user_prompt = state["messages"][-1].content if state.get("messages") else ""
    log = state.get("decision_log", [])
    if retriever:
        try:
            docs = retriever.invoke(user_prompt)
            context = "\n".join([doc.page_content for doc in docs])
            log.append("[RAG] ✅ Konteks SOP didapat dari ChromaDB.")
            return {"retrieved_context": context, "decision_log": log}
        except Exception as e:
            log.append(f"[RAG] ⚠️ Gagal retrieve: {e}")
    else:
        log.append("[RAG] ⚠️ Retriever tidak tersedia, skip.")
    return {"retrieved_context": "", "decision_log": log}


def node_agent1(state: AgentState) -> AgentState:
    """
    Agent 1: Verifikasi Kelayakan Akademik.
    Cek: (1) mahasiswa valid, (2) H-3 terpenuhi, (3) jadwal tidak bentrok.
    Implementasi rumus approval tunggal.
    """
    log = state.get("decision_log", [])
    log.append("─── [Agent 1: Verifikasi Akademik] ───")

    result = decide_approval(
        nim=state["student_id"],
        ruang=state["room_id"],
        tgl_kegiatan_str=state["date"],
        mulai=state["time_start"],
        selesai=state["time_end"],
    )

    log.append(f"  Mahasiswa: {result['student_check']['reason']}")
    log.append(f"  H-3: {result['h3_check']['reason']}")
    log.append(f"  Bentrok: {result['bentrok_check']['reason']}")
    if result['bentrok_check'].get('warning'):
        log.append(f"  {result['bentrok_check']['warning']}")

    if result["approved"]:
        msg = (f"✅ PEMINJAMAN DISETUJUI\n"
               f"• {result['student_check']['reason']}\n"
               f"• {result['h3_check']['reason']}\n"
               f"• Ruang {state['room_id']}: {result['bentrok_check']['reason']}")
        log.append("  → Keputusan: DISETUJUI ✅")
    else:
        reasons = []
        if not result['student_check']['valid']:
            reasons.append(result['student_check']['reason'])
        if not result['h3_check']['ok']:
            reasons.append(result['h3_check']['reason'])
        if not result['bentrok_check']['available']:
            reasons.append(result['bentrok_check']['reason'])
        msg = f"❌ PEMINJAMAN DITOLAK\n• " + "\n• ".join(reasons)
        log.append("  → Keputusan: DITOLAK ❌")

    return {
        "student_check": result["student_check"],
        "h3_check": result["h3_check"],
        "bentrok_check": result["bentrok_check"],
        "approval_result": result["approved"],
        "message": msg,
        "decision_log": log,
    }


def node_agent2(state: AgentState) -> AgentState:
    """
    Agent 2: Operasional / Sarpras.
    Jika disetujui: alokasi inventaris + generate digital pass + work order.
    Jika ditolak karena bentrok: cari alternatif ruangan.
    """
    log = state.get("decision_log", [])
    log.append("─── [Agent 2: Operasional Sarpras] ───")

    if state["approval_result"]:
        # Disetujui → alokasi inventaris + digital pass
        inv = allocate_inventory(state.get("items_needed", []))
        if inv["all_available"]:
            raw = f"{state['room_id']}{state['date']}{state['student_id']}"
            digital_pass = f"KP-{hashlib.md5(raw.encode()).hexdigest()[:8].upper()}"
            log.append(f"  Inventaris: semua tersedia ✅")
            log.append(f"  Digital Pass: {digital_pass}")
            log.append(f"  Work Order: {inv['work_order']['work_order_id']}")
            msg = (f"{state['message']}\n\n"
                   f"🎟️ Digital Pass: **{digital_pass}**\n"
                   f"📋 Work Order: {inv['work_order']['work_order_id']}\n"
                   f"📍 Ruang: {state['room_id']}\n"
                   f"📅 Tanggal: {state['date']}\n"
                   f"🕐 Jam: {state['time_start']} - {state['time_end']}")
            if state.get("keperluan"):
                msg += f"\n📝 Keperluan: {state['keperluan']}"
            return {
                "work_order": inv["work_order"],
                "digital_pass": digital_pass,
                "message": msg,
                "decision_log": log,
            }
        else:
            log.append(f"  Inventaris: tidak lengkap ❌ — {inv['status']}")
            return {
                "message": f"{state['message']}\n\n⚠️ Catatan: Beberapa inventaris tidak tersedia: {inv['status']}",
                "work_order": inv["work_order"],
                "decision_log": log,
            }
    else:
        # Ditolak — cek apakah karena bentrok, kalau iya cari alternatif
        if not state.get("bentrok_check", {}).get("available", True):
            log.append(f"  Mencari ruangan alternatif (selain {state['room_id']})...")
            try:
                tgl = datetime.strptime(state["date"], "%Y-%m-%d").date()
                alt = find_alternative_room(state["room_id"], tgl, state["time_start"], state["time_end"])
                if alt:
                    log.append(f"  → Alternatif ditemukan: {alt} ✅")
                    return {
                        "suggested_alternative": alt,
                        "message": f"{state['message']}\n\n💡 Saran alternatif: Ruang **{alt}** tersedia di tanggal dan jam yang sama.",
                        "decision_log": log,
                    }
                else:
                    log.append("  → Tidak ada ruangan alternatif yang tersedia. ❌")
                    return {
                        "message": f"{state['message']}\n\n❌ Tidak ada ruangan alternatif yang tersedia di tanggal dan jam tersebut.",
                        "decision_log": log,
                    }
            except Exception:
                log.append("  → Gagal mencari alternatif (tanggal tidak valid).")
                return {"decision_log": log}
        else:
            log.append("  Ditolak bukan karena bentrok — tidak perlu cari alternatif.")
            return {"decision_log": log}


def node_compose_response(state: AgentState) -> AgentState:
    """
    [FASE 1c] Compose jawaban natural via LLM menggunakan retrieved_context + decision_log.
    Kalau LLM tidak tersedia, pakai message yang sudah ada (fallback).
    """
    log = state.get("decision_log", [])
    log.append("─── [Compose Response] ───")

    if llm_pipe is None:
        log.append("  LLM tidak tersedia, pakai pesan template.")
        return {"decision_log": log}

    try:
        checks_summary = "\n".join(state.get("decision_log", []))
        raw = llm_pipe(COMPOSE_PROMPT.format(
            context=state.get("retrieved_context", "")[:800],
            checks=checks_summary,
            decision=state.get("message", ""),
        ))[0]["generated_text"]

        composed = raw.strip()
        if composed:
            state_message = state.get("message", "")
            # Gabungkan: jawaban LLM + data teknis (digital pass, work order)
            # Pertahankan info teknis dari message asli
            technical_parts = []
            for line in state_message.split("\n"):
                if any(marker in line for marker in ["🎟️", "📋", "📍", "📅", "🕐", "📝", "💡"]):
                    technical_parts.append(line)

            final_message = composed
            if technical_parts:
                final_message += "\n\n" + "\n".join(technical_parts)

            log.append("  ✅ Jawaban natural berhasil disusun via LLM.")
            return {"message": final_message, "decision_log": log}
        else:
            log.append("  ⚠️ LLM mengembalikan teks kosong, pakai pesan template.")
            return {"decision_log": log}
    except Exception as e:
        log.append(f"  ⚠️ Gagal compose via LLM ({e}), pakai pesan template.")
        return {"decision_log": log}


def node_end(state: AgentState) -> AgentState:
    """Node akhir — tidak melakukan apa-apa."""
    return state


# --- Routing Functions ---

def after_input(state: AgentState) -> str:
    """Setelah Input: cek apakah data lengkap. Jika tidak, langsung ke end (tanpa validasi)."""
    if state.get("data_complete"):
        return "rag"
    return "end"


def after_agent1(state: AgentState) -> str:
    """Setelah Agent 1: selalu lanjut ke Agent 2 (baik disetujui maupun ditolak)."""
    return "agent2"


def after_agent2(state: AgentState) -> str:
    """Setelah Agent 2: lanjut ke compose."""
    return "compose"


# ==========================================
# BAGIAN F: COMPILE GRAPH
# ==========================================
# Arsitektur:
# input → [data lengkap?] → YA: rag → agent1 → agent2 → compose → end → END
#                          → TIDAK: end → END (message sudah berisi info data kurang)
builder = StateGraph(AgentState)
builder.add_node("input", node_input)
builder.add_node("rag", node_rag_retrieval)
builder.add_node("agent1", node_agent1)
builder.add_node("agent2", node_agent2)
builder.add_node("compose", node_compose_response)
builder.add_node("end", node_end)

builder.set_entry_point("input")
builder.add_conditional_edges("input", after_input, {"rag": "rag", "end": "end"})
builder.add_edge("rag", "agent1")
builder.add_conditional_edges("agent1", after_agent1, {"agent2": "agent2"})
builder.add_conditional_edges("agent2", after_agent2, {"compose": "compose"})
builder.add_edge("compose", "end")
builder.add_edge("end", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ==========================================
# BAGIAN G: ANTARMUKA STREAMLIT
# ==========================================
st.set_page_config(page_title="Sistem Peminjaman Ruangan Kampus", page_icon="🏫", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
.jadwal-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 4px solid #3b82f6;
}
.jadwal-card h4 {
    color: #93c5fd;
    margin: 0 0 8px 0;
    font-size: 0.95rem;
}
.jadwal-card .jam {
    color: #fbbf24;
    font-weight: 600;
    font-size: 0.85rem;
}
.jadwal-card .ruang {
    color: #a5f3fc;
    font-size: 0.85rem;
}
.jadwal-card .dosen {
    color: #d1d5db;
    font-size: 0.8rem;
    margin-top: 4px;
}
.hari-header {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
    margin: 16px 0 8px 0;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# --- Layout Hybrid: Dashboard + Chat ---
col_dashboard, col_chat = st.columns([3, 2])

with col_dashboard:
    st.title("Sistem Peminjaman Ruangan Kampus")

    # Dashboard jadwal kuliah (visual cards)
    st.subheader("Jadwal Kuliah Reguler")

    if not jadwal_kuliah_df.empty:
        ITEMS_PER_PAGE = 6  # jumlah card per halaman

        # Filter hari & ruangan (sejajar)
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            hari_list = ["Semua"] + sorted(jadwal_kuliah_df["HARI"].unique().tolist())
            hari_filter = st.selectbox("🔍 Filter hari:", hari_list, key="jadwal_filter")
        with fcol2:
            temp_jk = jadwal_kuliah_df if hari_filter == "Semua" else jadwal_kuliah_df[jadwal_kuliah_df["HARI"] == hari_filter]
            ruang_list = ["Semua"] + sorted(temp_jk["RUANG"].unique().tolist())
            ruang_filter = st.selectbox("🔍 Filter ruangan:", ruang_list, key="ruang_filter")

        jk = temp_jk
        if ruang_filter != "Semua":
            jk = jk[jk["RUANG"] == ruang_filter]

        # Reset halaman jika filter berubah
        filter_key = f"{hari_filter}_{ruang_filter}"
        if st.session_state.get("_jadwal_filter_key") != filter_key:
            st.session_state["_jadwal_page"] = 0
            st.session_state["_jadwal_filter_key"] = filter_key

        # Flatten data menjadi list item (sorted by HARI order)
        hari_order = ["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU"]
        hari_rank = {h: i for i, h in enumerate(hari_order)}
        jk_sorted = jk.copy()
        jk_sorted["_hari_rank"] = jk_sorted["HARI"].map(hari_rank).fillna(99)
        jk_sorted = jk_sorted.sort_values("_hari_rank").drop(columns=["_hari_rank"])
        all_items = list(jk_sorted.iterrows())

        total_items = len(all_items)
        total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        current_page = st.session_state.get("_jadwal_page", 0)
        current_page = min(current_page, total_pages - 1)

        # Info halaman
        start_idx = current_page * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
        page_items = all_items[start_idx:end_idx]

        st.caption(f"Menampilkan {start_idx+1}–{end_idx} dari {total_items} jadwal  •  Halaman {current_page+1}/{total_pages}")

        # Render cards halaman ini
        prev_hari = None
        cols = st.columns(2)
        col_idx = 0
        for _, row in page_items:
            hari = row.get("HARI", "")
            if hari != prev_hari:
                # Header hari baru — render full-width lalu reset kolom
                st.markdown(f'<div class="hari-header">📅 {hari}</div>', unsafe_allow_html=True)
                cols = st.columns(2)
                col_idx = 0
                prev_hari = hari
            with cols[col_idx % 2]:
                st.markdown(f"""
                <div class="jadwal-card">
                    <h4>📖 {row.get('MATA_KULIAH', '-')}</h4>
                    <div class="jam">🕐 {row.get('JAM', '-')}</div>
                    <div class="ruang">📍 Ruang {row.get('RUANG', '-')}</div>
                    <div class="dosen">👤 {row.get('DOSEN', '-')}</div>
                </div>
                """, unsafe_allow_html=True)
            col_idx += 1

        # Navigasi halaman
        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("⬅️ Sebelumnya", disabled=(current_page == 0), key="prev_page", use_container_width=True):
                st.session_state["_jadwal_page"] = current_page - 1
                st.rerun()
        with nav2:
            # Pilihan halaman langsung
            page_options = [f"Halaman {i+1}" for i in range(total_pages)]
            selected = st.selectbox("Ke halaman:", page_options, index=current_page, key="page_select", label_visibility="collapsed")
            new_page = page_options.index(selected)
            if new_page != current_page:
                st.session_state["_jadwal_page"] = new_page
                st.rerun()
        with nav3:
            if st.button("Selanjutnya ➡️", disabled=(current_page >= total_pages - 1), key="next_page", use_container_width=True):
                st.session_state["_jadwal_page"] = current_page + 1
                st.rerun()
    else:
        st.info("Data jadwal kuliah belum tersedia.")

with col_chat:
    chat_head1, chat_head2 = st.columns([3, 1])
    with chat_head1:
        st.subheader("🤖 Asisten AI Peminjaman")
        st.caption("Universitas AMIKOM Yogyakarta")
    with chat_head2:
        if st.button("🔄 Reset Chat", key="btn_reset_chat", use_container_width=True, help="Sesi baru / ganti peminjam"):
            st.session_state.messages = []
            st.session_state["session_context"] = {}
            st.session_state["last_decision_log"] = None
            st.session_state["last_work_order"] = None
            st.rerun()

    # --- Initialize Persistent Session Context ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_context" not in st.session_state:
        st.session_state["session_context"] = {
            "nim": "", "nama": "", "room_id": "", "date": "",
            "time_start": "", "time_end": "", "keperluan": ""
        }

    # Render Chat Container
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                "👋 **Selamat datang!** Saya asisten AI peminjaman ruangan kampus AMIKOM Yogyakarta.\n\n"
                "Silakan beritahu saya **Nama, NIM, Ruangan, Tanggal, dan Jam** kegiatan Anda. "
                "Anda dapat menyampaikannya secara bertahap atau langsung lengkap."
            )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ketik permintaan atau pertanyaan peminjaman ruangan Anda di sini..."
    )

    if user_input:
        user_lower = user_input.lower().strip()

        # Deteksi reset kata kunci
        cancel_words = ["batal", "cancel", "mulai ulang", "reset", "ulangi dari awal"]
        if any(w in user_lower for w in cancel_words):
            st.session_state.messages = []
            st.session_state["session_context"] = {}
            st.session_state["last_decision_log"] = None
            st.session_state["last_work_order"] = None
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sesi peminjaman telah dibersihkan. Silakan ajukan peminjaman ruangan baru kapanpun! 👋"
            })
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": user_input})

        ctx = st.session_state["session_context"]
        extracted_new = extract_entities_lightweight(user_input)

        # 1. Jika ada NIM baru yang BERBEDA dari NIM di konteks, ini peminjam baru → Reset konteks
        if extracted_new.get("nim") and ctx.get("nim") and extracted_new["nim"] != ctx["nim"]:
            ctx = {
                "nim": "", "nama": "", "room_id": "", "date": "",
                "time_start": "", "time_end": "", "keperluan": ""
            }

        # 2. Update konteks dengan data baru
        for k, v in extracted_new.items():
            if v:
                ctx[k] = v

        # 3. Simpan kembali konteks ter-update
        st.session_state["session_context"] = ctx

        # 4. Bangun prompt sintetis dari akumulasi konteks
        prompt_to_send = build_synthetic_prompt(ctx)
        if not prompt_to_send:
            prompt_to_send = user_input

        # 5. Invoke Graph
        initial_state = {
            "student_id": "", "student_name": "", "room_id": "", "date": "",
            "time_start": "", "time_end": "", "keperluan": "",
            "items_needed": [], "retrieved_context": "",
            "data_complete": None,
            "student_check": None, "h3_check": None, "bentrok_check": None,
            "approval_result": None, "digital_pass": "", "suggested_alternative": None,
            "work_order": None, "decision_log": [], "message": "",
            "messages": [HumanMessage(content=prompt_to_send)]
        }
        config = {"configurable": {"thread_id": f"sesi_{uuid.uuid4().hex[:8]}"}}

        try:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Memproses peminjaman..."):
                        final_state = graph.invoke(initial_state, config)

            balasan = final_state.get("message", "Tidak ada hasil.")

            # Jika data belum lengkap sama sekali (misal sapaan halo)
            if not final_state.get("data_complete") and not any([ctx["nim"], ctx["room_id"], ctx["date"], ctx["time_start"]]):
                response_text = (
                    "Halo! 👋 Saya asisten AI peminjaman ruangan AMIKOM Yogyakarta.\n\n"
                    "Saya bisa membantu Anda mengecek ketersediaan dan meminjam ruangan kelas. "
                    "Silakan sebutkan **Nama, NIM, Ruangan, Tanggal, dan Jam** kegiatan Anda ya!"
                )
            else:
                full_response = balasan
                if final_state.get("digital_pass"):
                    full_response += f"\n\n🎟️ **Digital Pass: {final_state['digital_pass']}**"
                    # Transaksi berhasil -> reset konteks agar siap transaksi berikutnya
                    st.session_state["session_context"] = {}
                if final_state.get("work_order"):
                    wo = final_state["work_order"]
                    full_response += f"\n📋 **Work Order: {wo.get('work_order_id', '')}**"

                response_text = full_response

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state["last_decision_log"] = final_state.get("decision_log", [])
            st.session_state["last_work_order"] = final_state.get("work_order")

        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Terjadi kesalahan pada sistem: {e}"})

        st.rerun()

    # --- Log proses terakhir ---
    if st.session_state.get("last_decision_log"):
        with st.expander("🔍 Log Proses Agent AI (terakhir)", expanded=False):
            for log_entry in st.session_state["last_decision_log"]:
                st.text(log_entry)
    if st.session_state.get("last_work_order"):
        with st.expander("📋 Detail Work Order (terakhir)", expanded=False):
            st.json(st.session_state["last_work_order"])
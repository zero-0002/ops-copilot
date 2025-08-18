from typing import Tuple, Dict
from langdetect import detect
from unidecode import unidecode
import re
# Kullanıcı prompt'undaki TR terimlerini şema/iş terminolojisine map ediyoruz.
# (Gerektikçe bu sözlüğü büyütebilirsin.)
TERM_MAP_TR_TO_SCHEMA = {
    # domain kelimeleri → English canonical
    "rötar": "delay",
    "gecikme": "delay",
    "kötü hava": "weather",
    "hava koşulu": "weather",
    "iade": "refund",
    "şikayet": "complaint",
    "kabin ekibi": "crew",
    "uçuş numarası": "flight_number",
    "uçuş": "flight",
    "uçak tipi": "aircraft_type",
    "mürettebat": "crew",
    # kategoriler → db enumerations
    "bagaj": "baggage",
    "hizmet": "service",
    "koltuk": "seat",
    "diğer": "other",
    "kategori": "category",
    "kategorilerin": "category",
    "kategorisi": "category",
    "şikayet": "complaint",
    "şikayetler": "complaints",
    "günlük": "daily",
    "gunluk": "daily",
    "trend": "trend",
    "eğilim": "trend",
}
import re, unicodedata

_TR_MAP = str.maketrans({
    "ç":"c","Ç":"c","ğ":"g","Ğ":"g","ı":"i","I":"i","İ":"i",
    "ö":"o","Ö":"o","ş":"s","Ş":"s","ü":"u","Ü":"u",
})




def detect_lang(text:str)->str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def normalize_text(text: str) -> str:
    s = unidecode(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)   # noktalama → boşluk
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _replace_term_boundaries(text, src, dst):
    return re.sub(rf"\b{re.escape(src)}\b", dst, text)

def map_terms_to_schema(query:str)->Tuple[str, Dict[str, str]]:
    """
    Query içindeki TR terimleri kaba bir şekilde canonical İngilizce anahtar sözcüklere map edilir.
    Bu *prompt router* ve NL2SQL rehberi içindir; tablo/kolon *isimlerini çevirmiyoruz*.
    """
    q_norm = normalize_text(query)
    applied = {}
    for tr, en in TERM_MAP_TR_TO_SCHEMA.items():
        tr_norm = normalize_text(tr)
        if tr_norm in q_norm:
            q_norm = _replace_term_boundaries(q_norm, tr_norm, en)

            applied[tr] = en
    return q_norm, applied

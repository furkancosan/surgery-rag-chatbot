import os

"""
config.py
---------
Bu dosya, RAG chatbot projesinde kullanılan sabit ayarları tutar.

Bu versiyonda:
- Başlangıçta yüklenen sabit bir referans PDF YOK.
- Tüm içerik, kullanıcıların uygulama sırasında yüklediği PDF'lerden geliyor.
"""

# --- Dosya ve Dizin Yolları ---

# Artık başlangıçta sabit bir PDF kullanmıyoruz.
# İstersen ileride default bir PDF eklemek için tekrar doldurabilirsin.
PDF_YOLU = ""  # Kullanılmıyor

# FAISS indeks dosyalarının tutulacağı klasör adı
# (Eski isim INDEX_YOLU; geriye dönük uyumluluk için koruyoruz)
INDEX_YOLU = "faiss_index_saglik"

# Yeni: Persist edilen FAISS indeks klasörü
# Tüm load/save işlemlerinde bunu kullanacağız.
# INDEX_DIR'yi INDEX_YOLU ile aynı yaparak tek klasör üzerinden gideceğiz.
INDEX_DIR = INDEX_YOLU


# --- LangChain RAG Ayarları ---

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 0
RETRIEVER_K = 5

# Not: RERANK_TOP_K ileride retriever_k slider'ı ile birlikte
# doğrudan override edilebilir, ama şimdilik sabit dursun.
RERANK_TOP_K = 3

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Türkçe / İngilizce Pipeline Ayarları ---

USE_TURKISH_PIPELINE = True
TRANSLATION_MODEL = "llama-3.1-8b-instant"

# --- RAG Güven Seviyeleri ---

CONFIDENCE_HIGH = 0.50
CONFIDENCE_MEDIUM = 0.30

# --- Kapanış Önerisi (konuşmayı devam ettirme) ---

ENABLE_CLOSING_SUGGESTION = True

# --- Debug Ayarı ---

DEBUG = True

# --- Sistem İstemi (Yüksek güvendeki saf RAG için) ---

SYSTEM_PROMPT = (
    "Sen bir uzman Plastik Cerrahi bilgi asistanısın. Yalnızca sana sağlanan 'bağlam' içindeki "
    "bilgilere dayanarak kullanıcının sorusunu doğru ve özlü bir şekilde yanıtla. Bağlamda bilgi yoksa, "
    "'Referans dokümanımda bu konuda bilgi bulunmamaktadır.' şeklinde cevap ver ve tıbbi tavsiye vermekten KESİNLİKLE kaçın."
    "\n\nBağlam: {context}"
)

# app.py
"""
Web tabanlı Plastik Cerrahi RAG Chatbot arayüzü.

Özellikler:
- ChatGPT benzeri sohbet arayüzü (soru-cevaplar yukarı doğru birikir)
- Sol sidebar'da:
    - Üstte: Model Ayarları (temperature, retriever_k, max_tokens)
    - Hemen altında: PDF yükleme + indekse ekleme + doküman filtresi
- Arkada:
    - Kullanıcının yüklediği PDF'ler → FAISS vektör indeksi
    - Groq LLM + RAG zinciri
    - FAISS + kaynak doküman listesi disk üzerinde saklanır
"""

import os
import json
import hashlib
from pathlib import Path

import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import INDEX_YOLU, EMBEDDING_MODEL
from data_processor import load_and_chunk_data, create_vector_store, extend_vector_store
from rag_chain import setup_rag_chain
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# ============================================
# 1) Streamlit genel ayarları
# ============================================
st.set_page_config(
    page_title="Surgery RAG Chatbot",
    page_icon="🩺",
    layout="wide",
)


# ============================================
# 2) Yardımcı fonksiyonlar
# ============================================

def compute_file_hash(file_bytes: bytes) -> str:
    """
    Dosya içeriğinden MD5 hash üretir.
    Aynı içerikteki PDF'lerin tekrar indekse eklenmesini engellemek için kullanılır.
    (Sadece o oturum boyunca geçerlidir; disk üzerinde hash saklamıyoruz.)
    """
    return hashlib.md5(file_bytes).hexdigest()


def load_vector_store_from_disk():
    """
    Diskte daha önce kaydedilmiş bir FAISS vektör indeksi varsa yükler.

    Dönüş:
        - FAISS vector_store veya
        - None (indeks bulunamazsa veya hata oluşursa)
    """
    index_path = Path(INDEX_YOLU)

    if not index_path.exists():
        logger.info("FAISS indeks klasörü bulunamadı: %s", index_path)
        return None

    try:
        logger.info("Diskten FAISS vektör indeksi yükleniyor: %s", index_path)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS vektör indeksi başarıyla yüklendi.")
        return vector_store
    except Exception as e:
        logger.exception("FAISS indeksi yüklenirken hata oluştu: %s", e)
        return None


def save_vector_store_to_disk(vector_store):
    """
    Mevcut FAISS vektör indeksini diske kaydeder.
    """
    if vector_store is None:
        return

    index_path = Path(INDEX_YOLU)
    index_path.mkdir(parents=True, exist_ok=True)

    try:
        vector_store.save_local(str(index_path))
        logger.info("FAISS vektör indeksi diske kaydedildi: %s", index_path)
    except Exception as e:
        logger.exception("FAISS indeksi kaydedilirken hata oluştu: %s", e)


def load_available_sources():
    """
    Daha önce kaydedilmiş kaynak doküman listesini (dosya adlarını) diskten okur.

    Dönüş:
        - List[str] (doküman adları) veya boş liste
    """
    meta_path = Path(INDEX_YOLU) / "sources.json"
    if not meta_path.exists():
        logger.info("sources.json bulunamadı, boş liste ile başlanacak.")
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info("sources.json yüklendi. Doküman sayısı: %d", len(data))
            return data
        logger.warning("sources.json beklenen formatta değil, boş liste dönülecek.")
        return []
    except Exception as e:
        logger.exception("sources.json yüklenirken hata oluştu: %s", e)
        return []


def save_available_sources():
    """
    Mevcut available_sources listesini INDEX_YOLU içine sources.json olarak kaydeder.
    """
    meta_dir = Path(INDEX_YOLU)
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "sources.json"

    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                st.session_state["available_sources"],
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("available_sources metadata kaydedildi: %s", meta_path)
    except Exception as e:
        logger.exception("available_sources metadata kaydedilirken hata oluştu: %s", e)


def rebuild_rag_chain():
    """
    Mevcut vector_store + UI'deki model ayarlarına göre
    RAG zincirini (retrieval_chain) yeniden kurar.
    """
    vector_store = st.session_state["vector_store"]
    if vector_store is None:
        st.session_state["retrieval_chain"] = None
        logger.warning("RAG zinciri kurulamadı: vector_store None.")
        return

    temperature = st.session_state["temperature"]
    retriever_k = st.session_state["retriever_k"]
    max_tokens = st.session_state["max_tokens"]

    logger.info(
        "RAG zinciri yeniden kuruluyor (temperature=%.2f, k=%d, max_tokens=%d)",
        temperature,
        retriever_k,
        max_tokens,
    )

    retrieval_chain = setup_rag_chain(
        vector_store,
        temperature=temperature,
        retriever_k_override=retriever_k,
        max_tokens=max_tokens,
    )
    st.session_state["retrieval_chain"] = retrieval_chain
    logger.info("RAG zinciri başarıyla yeniden kuruldu.")


def add_uploaded_pdfs_to_index(uploaded_files):
    """
    Sidebar'dan yüklenen PDF dosyalarını:
    - Diske kaydeder
    - Chunk'lara böler
    - Eğer yeni içerikse FAISS indeksine ekler
    - Güncel FAISS indeksini ve kaynak doküman listesini diske kaydeder
    """
    if not uploaded_files:
        st.warning("Önce en az bir PDF seçmelisin.")
        return

    upload_dir = Path("uploaded_pdfs")
    upload_dir.mkdir(parents=True, exist_ok=True)

    all_new_docs = []
    total_new_docs = 0
    sources_changed = False

    for up_file in uploaded_files:
        file_bytes = up_file.getvalue()
        file_hash = compute_file_hash(file_bytes)

        # Aynı içerik daha önce indekslenmişse (bu oturumda) atla
        if file_hash in st.session_state["indexed_hashes"]:
            st.info(f"'{up_file.name}' içeriği zaten indekse eklenmiş, tekrar eklenmedi.")
            logger.info("Aynı içerik hash ile tespit edildi, atlandı: %s", up_file.name)
            continue

        # Yeni içerik → hash'i kaydet
        st.session_state["indexed_hashes"].add(file_hash)

        # Dosyayı diske yaz
        file_path = upload_dir / up_file.name
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        logger.info("Yeni PDF yüklendi ve kaydedildi: %s", file_path)

        # PDF'i chunk'lara böl
        docs_new = load_and_chunk_data(str(file_path), source_name=up_file.name)
        if docs_new:
            all_new_docs.extend(docs_new)
            total_new_docs += len(docs_new)

            if up_file.name not in st.session_state["available_sources"]:
                st.session_state["available_sources"].append(up_file.name)
                sources_changed = True
        else:
            logger.warning("PDF'den doküman üretilemedi: %s", file_path)

    if not all_new_docs:
        st.warning("Yüklenen dosyalardan yeni içerik eklenmedi (hepsi daha önce indekslenmiş olabilir).")
        return

    # Yeni dokümanları FAISS indeksine ekle / oluştur
    if st.session_state["vector_store"] is None:
        vector_store = create_vector_store(all_new_docs)
    else:
        vector_store = extend_vector_store(st.session_state["vector_store"], all_new_docs)

    if vector_store is None:
        st.error("Vektör deposu oluşturulamadı/güncellenemedi.")
        return

    st.session_state["vector_store"] = vector_store

    # Vector store değişti → diske kaydet + RAG zincirini yeniden kur
    save_vector_store_to_disk(vector_store)

    if sources_changed:
        save_available_sources()

    rebuild_rag_chain()

    st.success(f"✅ Yeni dokümanlar indekse eklendi. Toplam eklenen parça sayısı: {total_new_docs}")
    logger.info(
        "Yeni dokümanlar indekse eklendi. Toplam yeni parça: %d, available_sources: %s",
        total_new_docs,
        st.session_state["available_sources"],
    )


def render_assistant_meta(meta: dict):
    """
    Assistant cevabının altına RAG meta bilgisini (confidence, mode, sources vs.)
    profesyonel ve sade bir şekilde gösterir.
    """
    if not meta:
        return

    confidence = meta.get("confidence")
    mode = meta.get("mode")
    pages = meta.get("pages", [])
    sources = meta.get("sources", [])
    source_count = meta.get("source_count")

    with st.expander("📊 Cevap Özeti (RAG)", expanded=False):
        if confidence is not None:
            pct = int(confidence * 100)
            if confidence >= 0.7:
                label = "Yüksek güven"
                icon = "🟢"
            elif confidence >= 0.4:
                label = "Orta güven"
                icon = "🟡"
            else:
                label = "Düşük güven"
                icon = "🔴"
            st.markdown(f"**RAG Güven Skoru:** {icon} {pct}% ({label})")

        if mode:
            if mode == "pdf_strong":
                mode_text = "Cevap büyük oranda referans PDF içeriğine dayanıyor."
            elif mode == "hybrid":
                mode_text = "Cevap referans PDF + genel tıbbi bilgiyi birlikte kullanıyor."
            elif mode == "general":
                mode_text = "Referans PDF yetersiz, cevap daha çok genel tıbbi bilgiye dayanıyor."
            elif mode == "no_docs":
                mode_text = "İndekste henüz referans doküman yok."
            else:
                mode_text = ""
            st.markdown(f"**Çalışma Modu:** `{mode}`  \n{mode_text}")

        if pages:
            human_pages = [p + 1 for p in pages if isinstance(p, int)]
            st.markdown(f"**Kaynak sayfalar (0-based index):** {', '.join(map(str, pages))}")
            if human_pages:
                st.markdown(f"**Kaynak sayfalar (PDF sayfa numarası):** {', '.join(map(str, human_pages))}")

        if source_count is not None:
            st.markdown(f"**Kullanılan kaynak parça sayısı:** {source_count}")

        if sources:
            st.markdown("**Kaynak dokümanlar:**")
            for s in sources:
                name = s.get("name")
                pgs = s.get("pages", [])
                if name:
                    if pgs:
                        st.markdown(f"- `{name}` → sayfalar: {', '.join(map(str, pgs))}")
                    else:
                        st.markdown(f"- `{name}`")


# ============================================
# 3) Session state başlangıç değerleri
#    (helper fonksiyonlardan yararlanarak)
# ============================================

# Model ince ayarları (fine-tuning parametreleri)
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.0  # config'teki defaultunla eşlenebilir

if "retriever_k" not in st.session_state:
    st.session_state["retriever_k"] = 5

if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 768  # makul bir başlangıç değeri

# FAISS vektör indeksi
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = load_vector_store_from_disk()

# RAG zinciri
if "retrieval_chain" not in st.session_state:
    if st.session_state["vector_store"] is not None:
        # Vector store diskte vardı → açılışta direkt RAG zinciri kur
        rebuild_rag_chain()
    else:
        st.session_state["retrieval_chain"] = None

# Yüklü doküman isimleri (diskten oku)
if "available_sources" not in st.session_state:
    st.session_state["available_sources"] = load_available_sources()

# Aynı PDF içeriklerini oturum içinde yakalamak için hash set'i
if "indexed_hashes" not in st.session_state:
    st.session_state["indexed_hashes"] = set()

# Sohbet geçmişi
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ============================================
# 4) Sidebar: Ayarlar + Doküman yönetimi
# ============================================

with st.sidebar:
    st.markdown("### ⚙️ Model Ayarları")

    st.session_state["temperature"] = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["temperature"]),
        step=0.05,
        help="0.0 → daha deterministik, 1.0 → daha yaratıcı/dağınık.",
    )

    st.session_state["retriever_k"] = st.slider(
        "Retriever k",
        min_value=1,
        max_value=10,
        value=int(st.session_state["retriever_k"]),
        step=1,
        help="Sorgu başına FAISS'ten çekilecek maksimum parça sayısı.",
    )

    st.session_state["max_tokens"] = st.slider(
        "max_tokens",
        min_value=128,
        max_value=2048,
        value=int(st.session_state["max_tokens"]),
        step=64,
        help="Modelin üreteceği maksimum token sayısı.",
    )

    if st.button("Ayarları Uygula", type="primary", use_container_width=True):
        if st.session_state["vector_store"] is None:
            st.warning("Önce en az bir PDF yükleyip indekse eklemelisin.")
        else:
            rebuild_rag_chain()
            st.success("Model ayarları güncellendi ve RAG zinciri yeniden kuruldu.")

    st.markdown("---")
    st.markdown("### 📁 Kaynak Dokümanlar")

    uploaded_files = st.file_uploader(
        "PDF yükle",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("İndekse ekle ➕", use_container_width=True):
        if not uploaded_files:
            st.warning("Önce en az bir PDF seçmelisin.")
        else:
            # Yükleme + indeksleme sırasında spinner göster
            with st.spinner("Dokümanlar indekse ekleniyor..."):
                add_uploaded_pdfs_to_index(uploaded_files)

    st.markdown("**Yüklü dokümanlar:**")
    if st.session_state["available_sources"]:
        for name in st.session_state["available_sources"]:
            st.markdown(f"- `{name}`")
    else:
        st.caption("Henüz doküman yüklenmedi.")

    st.markdown("---")

    # Cevapta hangi dokümanların kullanılabileceğini seç (allowed_sources)
    allowed_sources = st.multiselect(
        "Cevaplarda kullanılacak dokümanlar",
        options=st.session_state["available_sources"],
        default=st.session_state["available_sources"],
        help="Boş bırakırsan tüm dokümanlar kullanılabilir.",
    )


# ============================================
# 5) Ana bölüm: Başlık + Chat arayüzü
# ============================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🩺 Surgery RAG Chatbot")
st.caption("Yüklediğin cerrahi PDF'ler üzerinde çalışan, Groq destekli soru-cevap asistanı.")

# Önce geçmiş mesajları render et
for msg in st.session_state["chat_history"]:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    meta = msg.get("meta", {})

    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and meta:
            render_assistant_meta(meta)

# En altta chat input
user_input = st.chat_input("Sorunu buraya yaz (Türkçe veya İngilizce)...")

if user_input:
    # 1) Kullanıcı mesajını kaydet ve göster
    st.session_state["chat_history"].append(
        {"role": "user", "content": user_input, "meta": {}}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Assistant cevabını üret
    retrieval_chain = st.session_state["retrieval_chain"]

    # allowed_sources boşsa None gönder → rag_chain tüm dokümanlardan seçsin
    allowed_sources_param = allowed_sources or None

    if retrieval_chain is None:
        # Henüz indeks yok veya RAG kurulmamış
        assistant_text = (
            "Henüz herhangi bir PDF indekse eklenmemiş görünüyor. "
            "Sol taraftan bir veya daha fazla PDF yükleyip 'İndekse ekle' butonuna bastıktan sonra soru sorabilirsin."
        )
        meta = {"mode": "no_docs", "confidence": 0.0}
    else:
        try:
            resp = retrieval_chain.invoke(
                {
                    "input": user_input,
                    "allowed_sources": allowed_sources_param,
                }
            )

            # Beklenen format:
            # {
            #   "answer": str,
            #   "pages": [...],
            #   "confidence": float,
            #   "source_count": int,
            #   "mode": "...",
            #   "sources": [...]
            # }
            if isinstance(resp, dict):
                answer = resp.get("answer", "")
                pages = resp.get("pages", [])
                confidence = resp.get("confidence", 0.0)
                mode = resp.get("mode", "none")
                source_count = resp.get("source_count", 0)
                sources = resp.get("sources", [])
            else:
                answer = str(resp)
                pages = []
                confidence = 0.0
                mode = "none"
                source_count = 0
                sources = []

            assistant_text = answer
            meta = {
                "pages": pages,
                "confidence": confidence,
                "mode": mode,
                "source_count": source_count,
                "sources": sources,
            }

        except Exception as e:
            logger.exception("Soru işlenirken hata oluştu: %s", e)
            assistant_text = (
                "Soru işlenirken bir hata oluştu. Logları kontrol etmen gerekebilir.\n\n"
                f"Hata: {e}"
            )
            meta = {"mode": "error", "confidence": 0.0}

    # 3) Assistant mesajını kaydet ve göster
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": assistant_text, "meta": meta}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_text)
        render_assistant_meta(meta)

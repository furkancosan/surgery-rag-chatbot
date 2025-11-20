# data_processor.py
"""
PDF yükleme, metin parçalama (chunking) ve FAISS vektör indeksi yönetimi.

Bu modül:

- load_and_chunk_data(pdf_path, source_name=None)
    → PDF dosyasını okur, chunk'lara böler ve her chunk'a
      source / source_name metadata'sını ekler.

- create_vector_store(docs)
    → Verilen doküman parçalarından yeni bir FAISS vektör deposu oluşturur.

- extend_vector_store(vector_store, new_docs)
    → Var olan FAISS vektör deposunu yeni dokümanlarla genişletir.

- save_vector_store(vector_store)
    → FAISS vektör indeksini diske kaydeder.

- load_vector_store()
    → Diskte FAISS indeks klasörü varsa yükler, yoksa None döner.

Not:
- INDEX_DIR, config.py içinde tanımlıdır ve FAISS indeks klasörünü temsil eder.
"""

import os
import logging
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, INDEX_DIR
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_and_chunk_data(pdf_path: str, source_name: Optional[str] = None):
    """
    PDF'yi yükler, metni parçalara ayırır ve her parçaya kaynak bilgisi (metadata) ekler.

    Parametreler:
        pdf_path: PDF dosyasının tam yolu (disk üzerindeki path)
        source_name: UI'de görünecek dosya adı.
                     - Örn: "Grabb_and_Smith_2019.pdf"
                     - None ise pdf_path içerisinden dosya adı alınır.

    Dönüş:
        docs: LangChain Document listesi (chunk'lanmış + metadata eklenmiş)
    """
    if not os.path.exists(pdf_path):
        logger.error("Belirtilen PDF yolu bulunamadı: %s", pdf_path)
        return None

    if source_name is None:
        source_name = os.path.basename(pdf_path)

    logger.info("PDF yükleniyor: %s (source_name=%s)", pdf_path, source_name)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info("Toplam %d sayfa yüklendi.", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(documents)
    logger.info("Parçalama sonrası toplam %d adet doküman parçası oluşturuldu.", len(docs))

    # Her parçaya kaynak metadata'sı ekle
    for doc in docs:
        if not isinstance(doc.metadata, dict):
            doc.metadata = {}

        # Orijinal PyPDFLoader metadata'sını koru, üstüne bizim alanlarımızı ekle
        doc.metadata.setdefault("file_path", pdf_path)
        doc.metadata["source"] = pdf_path          # uyumluluk için
        doc.metadata["source_name"] = source_name  # UI ve filtre için

    return docs


def _get_embeddings():
    """
    HuggingFace embedding modelini yükler (tek bir yerden).
    """
    logger.info("Yerel embedding modeli yükleniyor: %s", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings


def create_vector_store(docs):
    """
    Verilen doküman parçalarından yeni bir FAISS vektör deposu oluşturur.

    Parametreler:
        docs: LangChain Document listesi (chunk'lanmış)

    Dönüş:
        vector_store (FAISS) veya None
    """
    if not docs:
        logger.error("create_vector_store çağrıldı ama docs listesi boş.")
        return None

    embeddings = _get_embeddings()

    logger.info("Yeni FAISS vektör deposu oluşturuluyor (doküman sayısı: %d)...", len(docs))
    db = FAISS.from_documents(docs, embeddings)
    logger.info("FAISS vektör deposu başarıyla oluşturuldu.")
    return db


def extend_vector_store(vector_store, new_docs):
    """
    Var olan bir FAISS vektör deposunu yeni dokümanlarla genişletir.

    Parametreler:
        vector_store: Mevcut FAISS objesi
        new_docs: Eklenecek yeni LangChain Document listesi

    Dönüş:
        Güncellenmiş vector_store (aynı obje döner)
    """
    if vector_store is None:
        logger.warning("extend_vector_store: vector_store None, create_vector_store ile yenisi oluşturulacak.")
        return create_vector_store(new_docs)

    if not new_docs:
        logger.warning("extend_vector_store: new_docs boş, hiçbir doküman eklenmedi.")
        return vector_store

    logger.info(
        "FAISS vektör deposu yeni dokümanlarla genişletiliyor (eklenen parça sayısı: %d)...",
        len(new_docs),
    )
    vector_store.add_documents(new_docs)
    logger.info("FAISS vektör deposu başarıyla genişletildi.")
    return vector_store


def save_vector_store(vector_store):
    """
    FAISS vektör deposunu diske kaydeder.

    INDEX_DIR:
        - config.py içinde tanımlıdır.
        - Örn: 'faiss_index_saglik'
    """
    if vector_store is None:
        logger.warning("save_vector_store: vector_store None, kaydedilecek bir şey yok.")
        return

    os.makedirs(INDEX_DIR, exist_ok=True)
    logger.info("FAISS indeksi diske kaydediliyor: %s", INDEX_DIR)
    vector_store.save_local(INDEX_DIR)


def load_vector_store():
    """
    FAISS vektör deposunu diskten yüklemeye çalışır.

    Başarılı olursa:
        - FAISS objesini döner
    Aksi halde:
        - None döner
    """
    if not os.path.exists(INDEX_DIR):
        logger.info("Herhangi bir FAISS indeks klasörü bulunamadı: %s", INDEX_DIR)
        return None

    embeddings = _get_embeddings()
    logger.info("FAISS indeksi diskten yükleniyor: %s", INDEX_DIR)
    db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS indeksi başarıyla yüklendi.")
    return db

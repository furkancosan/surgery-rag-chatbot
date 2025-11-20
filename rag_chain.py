import os
import logging
from pathlib import Path
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import CrossEncoder

from config import (
    GROQ_MODEL,
    TEMPERATURE,
    SYSTEM_PROMPT,
    RETRIEVER_K,
    RERANK_TOP_K,  # hâlâ import ediyoruz; istersen ileride kaldırabilirsin
    RERANK_MODEL_NAME,
    USE_TURKISH_PIPELINE,
    TRANSLATION_MODEL,
    DEBUG,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    ENABLE_CLOSING_SUGGESTION,
)
from utils.logging_utils import setup_logger

"""
rag_chain.py
------------
Groq LLM + FAISS vektör indeksi + CrossEncoder reranker + (opsiyonel) TR/EN
çeviri pipeline'ını bir araya getirerek tam fonksiyonel bir RAG zinciri kurar.

Ana fonksiyon: setup_rag_chain(vector_store, ...)
Dönüş: RunnableLambda (invoke({...}) ile çağrılabilir)

Giriş formatı:
    {
        "input": "<kullanıcı sorusu>",
        "allowed_sources": ["doc1.pdf", "doc2.pdf"]  # opsiyonel (None = tüm dokümanlar)
    }

Çıkış formatı:
    {
        "answer": "<cevap + sohbet kapanışı>",
        "pages": [page_index_listesi (0-based)],
        "confidence": 0.0 - 1.0,
        "source_count": int,
        "mode": "pdf_strong" | "hybrid" | "general" | "none",
        "sources": [
            {"name": "doc1.pdf", "pages": [1, 2, 5]},
            {"name": "ek_dokuman.pdf", "pages": [3]}
        ]
    }
"""

logger = setup_logger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)

# Groq API anahtarı ortam değişkeninden okunur
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def is_turkish(text: str) -> bool:
    """
    Metinde Türkçe karakter olup olmadığını basitçe kontrol eder.

    Amaç:
    - Soru Türkçe mi İngilizce mi, hızlı ve hafif bir heuristik ile anlamak
    - Eğer Türkçe ise çeviri pipeline'ını devreye sokmak
    """
    tr_chars = "çğıöşüÇĞİÖŞÜ"
    return any(c in tr_chars for c in text)


def setup_rag_chain(
    vector_store,
    temperature: float | None = None,
    retriever_k_override: int | None = None,
    max_tokens: int | None = None,
):
    """
    Groq + FAISS ile RAG zincirini kurar.

    Parametreler:
        vector_store: FAISS vektör deposu
        temperature: (opsiyonel) LLM için sıcaklık. None ise config.TEMPERATURE kullanılır.
        retriever_k_override: (opsiyonel) FAISS'ten çekilecek parça sayısı. None ise config.RETRIEVER_K.
        max_tokens: (opsiyonel) LLM'in üreteceği maksimum token sayısı.

    Not:
        - effective_retriever_k hem FAISS'ten çekilen aday sayısını, hem de
          CrossEncoder sonrası LLM'e gönderilecek nihai doküman sayısını sınırlar.
    """
    if vector_store is None:
        raise ValueError(
            "Vektör deposu (vector_store) None geldi. "
            "Önce indeksin başarıyla yüklendiğinden/oluşturulduğundan emin olun."
        )

    if not GROQ_API_KEY:
        raise RuntimeError(
            "Groq API anahtarı bulunamadı. Lütfen 'GROQ_API_KEY' ortam değişkenini ayarlayın."
        )

    effective_temperature = temperature if temperature is not None else TEMPERATURE
    effective_retriever_k = (
        retriever_k_override if retriever_k_override is not None else RETRIEVER_K
    )

    logger.info(
        "RAG zinciri kuruluyor (Model: %s, temperature=%.2f, retriever_k=%d, max_tokens=%s)...",
        GROQ_MODEL,
        effective_temperature,
        effective_retriever_k,
        str(max_tokens),
    )

    # 1) Ana LLM (hem RAG cevabı hem de kapanış önerisi için kullanılacak)
    llm_main_kwargs: Dict[str, Any] = {}
    if max_tokens is not None:
        llm_main_kwargs["max_tokens"] = max_tokens

    llm_main = ChatGroq(
        temperature=effective_temperature,
        model_name=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        **llm_main_kwargs,
    )

    # 2) Reranker Modelini Yükleme (CrossEncoder)
    logger.info("Reranker CrossEncoder modeli yükleniyor: %s", RERANK_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    # 3) Çeviri LLM'i (Türkçe <-> İngilizce)
    if USE_TURKISH_PIPELINE:
        llm_translate = ChatGroq(
            temperature=0,
            model_name=TRANSLATION_MODEL or GROQ_MODEL,
            api_key=GROQ_API_KEY,
        )
        logger.debug(
            "Türkçe çeviri pipeline aktif. Model: %s",
            TRANSLATION_MODEL or GROQ_MODEL,
        )
    else:
        llm_translate = None
        logger.debug("Türkçe çeviri pipeline devre dışı.")

    # --- Çeviri yardımcı fonksiyonları ---

    def translate_tr_to_en(question_tr: str) -> str:
        """
        Türkçe soruyu İngilizceye çevirir.
        """
        if not llm_translate:
            return question_tr

        messages = [
            SystemMessage(
                content=(
                    "You are a professional translator. Translate the user's question "
                    "from Turkish to natural English suitable for a plastic surgery textbook context. "
                    "Only return the translated question, nothing else."
                )
            ),
            HumanMessage(content=question_tr),
        ]
        resp = llm_translate.invoke(messages)
        translated = getattr(resp, "content", str(resp)).strip()
        logger.debug("TR → EN çeviri tamamlandı.")
        return translated

    def translate_en_to_tr(answer_en: str) -> str:
        """
        İngilizce cevabı Türkçeye çevirir.
        """
        if not llm_translate:
            return answer_en

        messages = [
            SystemMessage(
                content=(
                    "You are a professional translator. Translate the following answer "
                    "from English to natural, fluent Turkish for a medical professional. "
                    "Only return the translated text, nothing else."
                )
            ),
            HumanMessage(content=answer_en),
        ]
        resp = llm_translate.invoke(messages)
        translated = getattr(resp, "content", str(resp)).strip()
        logger.debug("EN → TR çeviri tamamlandı.")
        return translated

    # --- Sohbet tarzı kapanış cümlesi üretici ---

    def generate_closing_note(
        original_question: str,
        final_answer: str,
        mode: str,
        original_is_turkish: bool,
    ) -> str:
        """
        Cevabın sonuna eklenecek, sohbeti devam ettiren 1-2 cümle üretir.
        """
        if not ENABLE_CLOSING_SUGGESTION:
            return ""

        lang_instr = (
            "Cümleyi Türkçe yaz."
            if original_is_turkish
            else "Write the sentences in English."
        )

        mode_info = {
            "pdf_strong": "The answer was mainly based on the reference textbook excerpt.",
            "hybrid": "The answer combined the reference textbook and general medical knowledge.",
            "general": "The answer was based on general medical knowledge, not the textbook.",
        }.get(mode, "")

        sys_msg = (
            "You are helping a plastic surgery learner in an ongoing chat. "
            "Based on the user's question and your answer, write 1-2 short sentences that:\n"
            "- Invite the user to continue the conversation,\n"
            "- Optionally suggest one or two related subtopics you could explain next,\n"
            "- Sound natural and friendly, not like a bullet list.\n"
            f"{lang_instr}\n\n"
            "Do not repeat the full answer. Do not list multiple questions; just write a small closing message.\n"
            f"Context about how the answer was generated: {mode_info}"
        )

        messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(
                content=f"Kullanıcının sorusu:\n{original_question}\n\nVerilen cevap:\n{final_answer}"
            ),
        ]

        resp = llm_main.invoke(messages)
        text = getattr(resp, "content", str(resp)).strip()
        logger.debug("Kapanış önerisi üretildi.")
        return text

    # --- Asıl RAG fonksiyonu ---

    def rag_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ana RAG işlevi.

        inputs:
            - input: str (zorunlu)
            - allowed_sources: List[str] | None (opsiyonel)

        allowed_sources:
            - None veya []         → tüm dokümanlar kullanılabilir
            - ["Grabb.pdf", ...]   → sadece bu source_name'lere sahip chunk'lar kullanılacak
        """
        question_raw = inputs.get("input", "").strip()
        allowed_sources = inputs.get("allowed_sources")  # List[str] veya None

        if not question_raw:
            logger.warning("Boş soru alındı.")
            return {
                "answer": "Soru boş geldi.",
                "pages": [],
                "confidence": 0.0,
                "source_count": 0,
                "mode": "none",
                "sources": [],
            }

        logger.debug("Allowed_sources: %s", allowed_sources)

        # 1) Dil tespiti & gerekirse EN'e çeviri
        original_is_turkish = USE_TURKISH_PIPELINE and is_turkish(question_raw)

        if original_is_turkish:
            logger.debug("Soru Türkçe algılandı, İngilizceye çevriliyor...")
            question_en = translate_tr_to_en(question_raw)
        else:
            question_en = question_raw

        logger.debug("RAG için kullanılacak soru (EN): %s", question_en)

        # 2) FAISS ile benzerlik + skor (confidence)
        try:
            results = vector_store.similarity_search_with_relevance_scores(
                question_en,
                k=effective_retriever_k,
            )
        except AttributeError:
            logger.debug(
                "similarity_search_with_relevance_scores bulunamadı, fallback'e geçiliyor."
            )
            docs_only = vector_store.similarity_search(
                question_en, k=effective_retriever_k
            )
            results = [(doc, 0.0) for doc in docs_only]

        docs = [doc for doc, _ in results]
        scores = [float(score) for _, score in results] if results else []
        confidence = max(scores) if scores else 0.0

        logger.debug(
            "FAISS'ten %d doküman döndü. Max skor (confidence): %.3f",
            len(docs),
            confidence,
        )

        # 2.5) CrossEncoder ile reranking
        reranked_docs = docs
        if docs:
            try:
                pairs = [[question_en, doc.page_content] for doc in docs]
                rerank_scores = reranker.predict(pairs)
                doc_score_pairs = list(zip(docs, rerank_scores))

                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                # 🔹 Burada artık RERANK_TOP_K yerine effective_retriever_k kullanıyoruz
                top_k = min(effective_retriever_k, len(doc_score_pairs))
                reranked_docs = [doc for doc, _ in doc_score_pairs[:top_k]]

                logger.debug(
                    "Reranker: İlk FAISS sonuç sayısı: %d, top_k (LLM'e giden parça sayısı): %d",
                    len(docs),
                    top_k,
                )
                for i, (doc, s) in enumerate(doc_score_pairs[:top_k], start=1):
                    page = (
                        doc.metadata.get("page")
                        if isinstance(doc.metadata, dict)
                        else None
                    )
                    logger.debug("  %d. skor=%.3f (page=%s)", i, s, page)
            except Exception as e:
                logger.warning("Reranking sırasında hata oluştu: %s", e)

        # 2.6) allowed_sources filtresini uygula (varsa)
        if allowed_sources:
            allowed_set = set(allowed_sources)
            filtered_docs = []
            for doc in reranked_docs:
                md = doc.metadata if isinstance(doc.metadata, dict) else {}
                src_name = md.get("source_name") or md.get("source")
                if src_name and Path(src_name).name in allowed_set:
                    filtered_docs.append(doc)

            logger.debug(
                "allowed_sources filtresi uygulandı. Önce: %d doküman, sonra: %d doküman.",
                len(reranked_docs),
                len(filtered_docs),
            )
            reranked_docs = filtered_docs

            # Hiç doküman kalmazsa, direkt bilgilendirici cevap dön
            if not reranked_docs:
                return {
                    "answer": (
                        "Seçtiğin doküman(lar) içinde bu soruya dair yeterli bilgi bulamadım. "
                        "İstersen tüm dokümanlara açık şekilde tekrar deneyebilirsin."
                    ),
                    "pages": [],
                    "confidence": 0.0,
                    "source_count": 0,
                    "mode": "none",
                    "sources": [],
                }

        # 3) Mode belirleme (confidence FAISS skorlarından geliyor)
        if confidence >= CONFIDENCE_HIGH:
            mode = "pdf_strong"
        elif confidence >= CONFIDENCE_MEDIUM:
            mode = "hybrid"
        else:
            mode = "general"

        # 4) Sayfa numaralarını topla (0-based)
        page_numbers = sorted(
            {
                doc.metadata.get("page")
                for doc in reranked_docs
                if isinstance(doc.metadata, dict) and "page" in doc.metadata
            }
        )

        # 4.1) Kaynak doküman + sayfalar bilgisini çıkar
        sources_dict: Dict[str, set[int]] = {}
        for doc in reranked_docs:
            md = doc.metadata if isinstance(doc.metadata, dict) else {}
            raw_source_name = md.get("source_name") or md.get("source")

            if not raw_source_name:
                continue

            source_name = Path(raw_source_name).name
            page_idx = md.get("page")

            if isinstance(page_idx, int):
                human_page = page_idx + 1
            else:
                human_page = None

            if source_name not in sources_dict:
                sources_dict[source_name] = set()

            if human_page is not None:
                sources_dict[source_name].add(human_page)

        sources_list = []
        for name, pages_set in sources_dict.items():
            pages_sorted = sorted(pages_set)
            sources_list.append(
                {
                    "name": name,
                    "pages": pages_sorted,
                }
            )

        context_text = "\n\n".join(doc.page_content for doc in reranked_docs)

        logger.debug("Reranker sonrası kullanılan doküman sayısı: %d", len(reranked_docs))
        logger.debug("Kaynak sayfalar (0-based index): %s", page_numbers)
        logger.debug("RAG çalışma modu: %s, confidence: %.3f", mode, confidence)
        logger.debug("Kaynak doküman sayısı: %d", len(sources_list))

        # 5) Mode'a göre system prompt ve context
        if mode == "pdf_strong":
            system_content = SYSTEM_PROMPT.format(context=context_text)

        elif mode == "hybrid":
            system_content = (
                "You are a plastic surgery educational assistant. You have access to:\n"
                "1) A reference textbook excerpt (called 'context'), and\n"
                "2) Your general medical knowledge.\n\n"
                "Use BOTH sources to answer the question. If there is any conflict, prioritize the textbook context. "
                "In your answer, clearly say that the information is based on both the reference document and general "
                "medical knowledge. Do NOT give personalized medical advice or treatment recommendations.\n\n"
                f"Context:\n{context_text}"
            )
        else:  # general
            system_content = (
                "You are a plastic surgery educational assistant. The reference document does not contain "
                "sufficiently reliable information to answer the user's question. Answer using ONLY your general "
                "medical knowledge at a high, educational level. Clearly state that this answer is NOT based on the "
                "provided reference document. Do NOT give clinical advice or treatment recommendations for individual "
                "patients."
            )

        # 6) Ana modele mesaj gönder (RAG cevabı için)
        messages_main = [
            SystemMessage(content=system_content),
            HumanMessage(content=question_en),
        ]

        resp = llm_main.invoke(messages_main)
        answer_en = getattr(resp, "content", str(resp)).strip()
        logger.debug("Ana LLM cevabı alındı.")

        # 7) Gerekirse cevabı Türkçeye çevir
        final_answer = (
            translate_en_to_tr(answer_en) if original_is_turkish else answer_en
        )

        # 8) Sohbet tarzı kapanış notu
        closing_note = generate_closing_note(
            original_question=question_raw,
            final_answer=final_answer,
            mode=mode,
            original_is_turkish=original_is_turkish,
        )

        if closing_note:
            final_answer_with_note = f"{final_answer}\n\n{closing_note}"
        else:
            final_answer_with_note = final_answer

        return {
            "answer": final_answer_with_note,
            "pages": page_numbers,          # 0-based index
            "confidence": round(confidence, 3),
            "source_count": len(reranked_docs),
            "mode": mode,
            "sources": sources_list,        # kaynak doküman listesi
        }

    retrieval_chain = RunnableLambda(rag_fn)

    logger.info(
        "RAG zinciri başarıyla kuruldu (TR/EN, 3-seviyeli confidence, reranker + sohbet kapanışlı, "
        "temperature=%.2f, retriever_k=%d, max_tokens=%s).",
        effective_temperature,
        effective_retriever_k,
        str(max_tokens),
    )
    return retrieval_chain

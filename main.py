import time
import os
import sys
import traceback
from pathlib import Path

from groq import Groq
try:
    # Groq SDK sürümlerinde farklı exception isimleri olabileceği için esnek import
    from groq import GroqError
except ImportError:
    GroqError = Exception

from utils.logging_utils import setup_logger

"""
main.py
-------
Bu dosya, RAG sistemini komut satırından (CLI) etkileşimli bir chatbot olarak çalıştırır.
"""

logger = setup_logger(__name__)

# Proje kök dizinini sys.path'e ekle (göreli import sorunlarını engellemek için)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PDF_YOLU, INDEX_YOLU
from data_processor import load_and_chunk_data, get_vector_store
from rag_chain import setup_rag_chain


def main():
    """
    Komut satırı tabanlı Plastik Cerrahi RAG chatbot'unu başlatır.
    """
    logger.info("Plastik Cerrahi Chatbot (Groq/LangChain RAG) başlatılıyor...")
    print("--- 🩺 Plastik Cerrahi Chatbot (Groq/LangChain RAG) Başlatılıyor ---")

    # 1. Veriyi hazırlama (indeks yoksa PDF'ten parçalar oluştur)
    docs = None
    if not os.path.exists(INDEX_YOLU):
        logger.info("FAISS indeks bulunamadı, PDF'ten yeni indeks oluşturulacak.")
        docs = load_and_chunk_data(PDF_YOLU)

    # 2. Vektör deposu (FAISS) oluşturma / yükleme
    vector_store = get_vector_store(docs)
    if not vector_store:
        logger.error("Vektör deposu oluşturulamadı/yüklenemedi.")
        print("❌ Uygulama başlatılamadı. Vektör deposu oluşturulamadı/yüklenemedi.")
        return

    # 3. RAG zincirini kurma
    retrieval_chain = setup_rag_chain(vector_store)

    print("\n--- Chatbot Hazır: Sorgulama Yapmaya Başlayın ---")
    print("Çıkmak için 'exit', 'quit' veya 'çık' yazın.")

    # 4. Kullanıcı etkileşim döngüsü
    while True:
        soru = input("\nSoru: ")
        if soru.lower() in ["exit", "quit", "çık"]:
            logger.info("Kullanıcı çıkış komutu verdi, chatbot sonlandırılıyor.")
            print("Chatbot sonlandırılıyor. İyi günler! 👋")
            break

        if not soru.strip():
            # Boş string girildiyse yeniden isteme
            continue

        logger.info("Yeni soru alındı: %s", soru)
        start_time = time.time()

        try:
            # Zinciri çalıştır (RAG cevabı al)
            response = retrieval_chain.invoke({"input": soru})
            end_time = time.time()

            answer = None
            pages = None
            confidence = None
            mode = None
            sources = None

            if isinstance(response, str):
                answer = response
            elif isinstance(response, dict):
                answer = response.get("answer", "")
                pages = response.get("pages", None)
                confidence = response.get("confidence", None)
                mode = response.get("mode", None)
                sources = response.get("sources", [])
            else:
                answer = str(response)
                logger.warning("Yanıt beklenmedik formatta geldi: %s", type(response))
                print("UYARI: Yanıt beklenmedik formatta geldi. Ham çıktı yazdırılıyor.")

            # Sonucu yazdır
            print("\n🤖 Yanıt:")
            print(answer)

            # Referans sayfalar
            if pages:
                human_pages = [p + 1 for p in pages if isinstance(p, int)]
                if human_pages:
                    print(f"\n📄 Referans sayfalar (PDF index): {', '.join(map(str, human_pages))}")

            # Kaynak dokümanlar
            if sources:
                print("\n📚 Kaynak dokümanlar:")
                for src in sources:
                    name = src.get("name", "Bilinmeyen doküman")
                    pgs = src.get("pages", [])
                    if pgs:
                        print(f"  - {name} (Sayfalar: {', '.join(map(str, pgs))})")
                    else:
                        print(f"  - {name}")

            # Confidence ve mod
            if confidence is not None:
                print(f"\n🔎 RAG güven skoru: {confidence:.2f}")
            if mode is not None:
                print(f"⚙️ Çalışma modu: {mode}")

            print(f"\n(Yanıt süresi: {end_time - start_time:.2f} saniye)")
            logger.info(
                "Soru yanıtlandı. Süre: %.2f sn, mode=%s, confidence=%.3f",
                end_time - start_time, mode, confidence or 0.0
            )

        except GroqError as e:
            status = getattr(e, "status_code", None)
            logger.error("GroqError yakalandı. Status=%s, Mesaj=%s", status, e)
            if status == 500:
                print("\n❌ Bir Hata Oluştu (Groq 500 - Internal Server Error):")
                print("Groq tarafında geçici bir sunucu hatası oluştu (HTTP 500).")
                print("Kodun ve isteğin büyük ihtimalle doğru; bu tür hatalar genelde servis tarafında kısa süreli sorunlardan kaynaklanır.")
            else:
                print("\n❌ Groq İstek Hatası:")
                print(f"Hata Mesajı: {e}")

        except Exception as e:
            logger.exception("Beklenmeyen bir hata oluştu: %s", e)
            print("\n❌ Bir Hata Oluştu:")
            print(f"Hata Mesajı: {e}")
            print("\n--- Detaylı Traceback ---")
            traceback.print_exc()
            print("\nOlası Nedenler:")
            print("* Groq API anahtarı (GROQ_API_KEY) yanlış, eksik veya limit aşımı.")
            print("* Vektör İndeksi (`faiss_index_saglik`) bozuk veya yüklenemedi.")
            print("* İnternet bağlantısı veya Groq servisine erişim sorunu.")


if __name__ == "__main__":
    main()

import logging
import os
from pathlib import Path

"""
logging_utils.py
----------------
Proje genelinde kullanılacak ortak logging katmanını içerir.

Amaç:
- Tüm modüller için tutarlı bir log formatı ve seviyesi sağlamak
- Hem konsola hem de isteğe bağlı olarak log dosyasına yazmak
- Her modülde ayrı ayrı logging ayarı yapmak yerine, tek bir yerden yönetmek
"""

# Ortam değişkenlerinden log seviyesi ve klasörü okunabilir
DEFAULT_LEVEL_NAME = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(os.getenv("RAG_LOG_DIR", "logs"))  # varsayılan: ./logs/


LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def get_log_level() -> int:
    """
    Ortam değişkenine göre log seviyesini döndürür.
    Geçersiz bir değer varsa INFO'ya düşer.
    """
    return LEVEL_MAP.get(DEFAULT_LEVEL_NAME, logging.INFO)


def setup_logger(name: str) -> logging.Logger:
    """
    Verilen isim için (genellikle __name__) bir logger döndürür.

    Özellikler:
    - Aynı isimle daha önce logger oluşturulmuşsa handler eklemez (çift log satırını engeller)
    - Konsola log yazar
    - logs/app.log dosyasına log yazar
    """
    logger = logging.getLogger(name)

    # Eğer logger zaten konfigüre edilmişse tekrar handler ekleme
    if logger.handlers:
        return logger

    logger.setLevel(get_log_level())

    # logs klasörünü oluştur
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Ortak format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1) Konsola yazan handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2) Dosyaya yazan handler
    file_path = LOG_DIR / "app.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Üst logger'lara tekrar köpürmesin
    logger.propagate = False

    logger.debug("Logger initialized with level %s", DEFAULT_LEVEL_NAME)
    return logger

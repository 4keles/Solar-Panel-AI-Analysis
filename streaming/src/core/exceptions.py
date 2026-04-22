"""Custom exceptions for the streaming project."""

class StreamingError(Exception):
    """Tüm streaming modülü hataları için taban sınıf."""
    pass

class ModelLoadError(StreamingError):
    """Model .pt dosyası bulunamadığında veya yüklenemediğinde fırlatılır."""
    pass

class ModelNotLoadedError(StreamingError):
    """Model yüklenmeden inference çalıştırılmak istendiğinde fırlatılır."""
    pass

class SourceOpenError(StreamingError):
    """Kamera veya veri kaynağı açılamadığında fırlatılır."""
    pass

class SourceNotOpenError(StreamingError):
    """Kaynak açılmadan okuma yapılmak istendiğinde fırlatılır."""
    pass

class ProcessorNotInitializedError(StreamingError):
    """İşlemci veya pipeline çalışmak için başlatılmamışken fırlatılır."""
    pass

class RecorderAlreadyRunningError(StreamingError):
    """Kayıt zaten aktifken kaydı yeniden başlatma denemesinde fırlatılır."""
    pass

class RecorderNotRunningError(StreamingError):
    """Kayıt aktif değilken kaydı durdurma istenirse fırlatılır."""
    pass

class RecorderSetupError(StreamingError):
    """Kayıt klasörü yaratılamazsa veya benzeri I/O disk kaynaklı bir hata olursa fırlatılır."""
    pass

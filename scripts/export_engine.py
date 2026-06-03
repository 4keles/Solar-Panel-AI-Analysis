#!/usr/bin/env python3
"""
Modeli TensorRT (.engine) formatına dönüştürmek için yardımcı betik.
Bu format, canlı akış (streaming) ve cihaz üzerinde maksimum performans (FPS) almak için kullanılır.

Kullanım:
    uv run scripts/export_engine.py --model models/v1.0.4/best.pt --imgsz 640
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Hata: 'ultralytics' kütüphanesi bulunamadı. Lütfen sanal ortamın aktif olduğundan emin olun.")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO modelini TensorRT Engine (.engine) formatına çevirir.")
    parser.add_argument("--model", type=Path, required=True, help="Eğitilmiş .pt modelinin yolu (Örn: models/v1.0.4/best.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Giriş boyutu (Varsayılan: 640)")
    parser.add_argument("--device", type=int, default=0, help="GPU cihaz numarası (Varsayılan: 0)")
    parser.add_argument("--no-half", action="store_true", help="FP16 (Half) precision kapatmak için kullanın.")
    
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Hata: Model dosyası bulunamadı: {args.model}")
        sys.exit(1)

    print(f"[*] Model yükleniyor: {args.model}")
    model = YOLO(args.model)
    
    print("[*] TensorRT (.engine) Export işlemi başlatılıyor...")
    print("    Not: 'tensorrt' paketleri ortamda eksikse Ultralytics otomatik olarak indirmeye çalışabilir.")
    print("    Bu işlem internet hızınıza bağlı olarak uzun sürebilir (yaklaşık 2 GB indirme yapabilir).")
    
    try:
        exported_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            device=args.device,
            half=not args.no_half,
            workspace=4  # TensorRT build için maksimum bellek sınırı (GB)
        )
        print(f"\n[+] BAŞARILI! Model dışa aktarıldı: {exported_path}")
        print("    Artık bu .engine dosyasını canlı yayın (streaming) modülünde kullanabilirsiniz.")
    except Exception as e:
        print(f"\n[-] EXPORT BAŞARISIZ OLDU. Hata detayı:\n{e}")
        print("\n[!] Çözüm: Ortamınızda TensorRT kurulu olmayabilir veya indirme kesilmiş olabilir.")
        print("Manuel olarak kurmak için terminalde şu komutu çalıştırın:")
        print("    uv pip install tensorrt-cu12 onnxruntime-gpu")
        print("Ardından bu betiği tekrar çalıştırın.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Active Learning Pipeline.

Faz 1: Auto-annotation and waiting for Label Studio verification.
Faz 2: Augmentation and dataset merging.
"""

import argparse
import subprocess
import time
from pathlib import Path

from scripts.auto_annotate import auto_annotate
from scripts.augment import augment_dataset
from scripts.merge_dataset import merge_dataset
from scripts.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="2-Phase Active Learning Pipeline")
    parser.add_argument("--source", type=Path, default=Path("data/raw_data/captured"), help="Capture klasörü")
    parser.add_argument("--model", type=Path, default=Path("models/latest/best.pt"), help="Auto-annotate için güncel model")
    parser.add_argument("--aug-config", type=Path, default=Path("configs/augmentation_drone.yaml"), help="Drone augmentation dosyası")
    parser.add_argument("--target", type=Path, default=Path("data/processed/finetune"), help="Birleştirilecek hedef dataset")
    parser.add_argument("--multiplier", type=int, default=8, help="1 resmi kaç katına çoğaltalım? (Örn: 8x)")
    args = parser.parse_args()

    logger.info("pipeline_started", phase=1, source=str(args.source))

    # ── Faz 1: Auto-Annotation ────────────────────────────────────────────────
    images = list(args.source.glob("*.jpg")) + list(args.source.glob("*.png"))
    if not images:
        logger.error("no_images_found", source=str(args.source))
        return

    logger.info("running_auto_annotation", count=len(images))
    auto_annotate(args.source, args.model, conf_threshold=0.25)
    logger.info("auto_annotation_finished")

    # ── Label Studio Kontrolü (User Review) ──────────────────────────────────
    print("\n" + "="*60)
    print("FAZ 1 TAMAMLANDI: Otomatik etiketler oluşturuldu.")
    print("="*60)
    print("Label Studio başlatılıyor... Lütfen tarayıcınızdan arayüze girip:")
    print(f"1. {args.source.absolute()} klasörünü projeye bağlayın/import edin.")
    print("2. Import ederken hedef klasördeki 'classes.txt' dosyasını göstermeyi unutmayın!")
    print("   Böylece sınıf sırası (Fiziksel hasar, kuş pisliği vs.) modelle birebir aynı kalır.")
    print("3. Hatalı veya eksik çizilen kutuları düzeltip kaydedin.")
    print("="*60 + "\n")

    ls_process = None
    try:
        # Try to launch label-studio in background via uv
        ls_process = subprocess.Popen(["uv", "run", "label-studio"])
        logger.info("label_studio_started_via_uv")
    except FileNotFoundError:
        logger.warning("uv_or_label_studio_not_found_in_path")
        print("Sistemde 'uv run label-studio' başlatılamadı. Lütfen kendiniz başlatın.")
    
    # Wait for user confirmation
    input(">>> Etiketleme işleminizi tamamen bitirdiğinizde ENTER tuşuna basarak FAZ 2'ye geçin... ")

    if ls_process:
        print("Label Studio arka planda kapatılıyor...")
        ls_process.terminate()

    # ── Sınıf Doğrulama (Validation) ──────────────────────────────────────────
    # Sınıf kaymalarını önlemek için max class ID kontrolü yapıyoruz.
    classes_file = args.source / "classes.txt"
    max_valid_class_id = -1
    if classes_file.exists():
        with open(classes_file, "r") as f:
            max_valid_class_id = len([line for line in f.read().splitlines() if line.strip()]) - 1

    valid_images = []
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        
        # Sınıf ID'lerini kontrol et
        is_valid = True
        with open(txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        if max_valid_class_id != -1 and cls_id > max_valid_class_id:
                            logger.error("invalid_class_id", file=str(txt), id=cls_id, max=max_valid_class_id)
                            print(f"HATA: {txt.name} dosyasında geçersiz bir sınıf numarası ({cls_id}) bulundu! Dosya atlanıyor.")
                            is_valid = False
                            break
                    except ValueError:
                        pass
        
        if is_valid:
            valid_images.append(img)

    # ── Faz 2: Augmentation & Formatting ──────────────────────────────────────
    target_aug_count = len(valid_images) * args.multiplier

    print(f"Tespit edilen doğrulanmış resim: {len(valid_images)}")
    print(f"Çarpan: {args.multiplier}x  -> Üretilecek toplam sentetik veri: {target_aug_count}")
    
    aug_output_dir = args.source / "augmented_temp"
    aug_output_dir.mkdir(parents=True, exist_ok=True)

    if target_aug_count > 0:
        logger.info("running_augmentation", target_count=target_aug_count)
        try:
            augment_dataset(
                source=args.source,
                output=aug_output_dir,
                pipeline_path=args.aug_config,
                target_count=target_aug_count
            )
        except Exception as e:
            logger.error("augmentation_failed", error=str(e))
            print("Augmentation sırasında bir hata oluştu. Pipeline durduruluyor.")
            return

    # ── Veri Birleştirme (Merge) ──────────────────────────────────────────────
    logger.info("merging_dataset", target=str(args.target))
    # We merge both the original captured folder and the augmented folder
    # 1. Merge augmented
    if list(aug_output_dir.rglob("*.jpg")):
        merge_dataset(source_dir=aug_output_dir, target_base=args.target, split_ratio=0.8)
    
    # 2. Merge originals (and archive them)
    merge_dataset(source_dir=args.source, target_base=args.target, split_ratio=0.8)

    # Cleanup temp augmented
    if aug_output_dir.exists():
        import shutil
        shutil.rmtree(aug_output_dir)

    print("\n" + "="*60)
    print("PIPELINE TAMAMLANDI! 🎉")
    print(f"Tüm veriler (orijinal + çoğaltılmış) {args.target} klasörüne taşındı.")
    print("Eğitime (Fine-Tuning) hazırsınız.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

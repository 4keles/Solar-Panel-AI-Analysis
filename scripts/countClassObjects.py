#!/usr/bin/env python3

#Bu script, bir dizindeki tüm txt dosyalarını tarar ve her bir satırın başındaki
#sınıf indeksine göre, verilen YAML dosyasındaki sınıf isimlerinin sayısını hesaplar.


import os
import sys
import argparse
import yaml
from collections import defaultdict
import glob


def read_yaml_classes(yaml_path):

    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        # 'names' anahtarını kontrol et
        if 'names' not in data:
            print(f"Hata: YAML dosyasında 'names' anahtarı bulunamadı: {yaml_path}")
            sys.exit(1)
        
        class_names = data['names']
        
        # Sınıf isimlerini indeksleriyle birlikte yazdır (kontrol için)
        print("YAML'dan okunan sınıflar:")
        for idx, name in enumerate(class_names):
            print(f"  {idx}: {name}")
        
        return class_names
    
    except FileNotFoundError:
        print(f"Hata: YAML dosyası bulunamadı: {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Hata: YAML dosyası okunurken bir hata oluştu: {e}")
        sys.exit(1)


def count_classes_in_directory(directory_path, class_names):
    """
    Dizindeki tüm txt dosyalarını tarar ve her sınıfın kaç kez geçtiğini sayar.
    """
    # Sınıf sayıları için sözlük (varsayılan olarak 0)
    class_counts = defaultdict(int)
    
    # Dizindeki tüm txt dosyalarını bul
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    
    if not txt_files:
        print(f"Uyarı: {directory_path} dizininde dosya bulunmadı.")
        return class_counts
    
    print(f"\nBulunan dosya:{len(txt_files)}")
    
    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        file_class_counts = defaultdict(int)
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Boş satırları atla
                    if not line:
                        continue
                    
                    # Satırı boşluklardan böl ve ilk elemanı al (sınıf indeksi)
                    parts = line.split()
                    if not parts:
                        continue
                    
                    try:
                        class_idx = int(parts[0])
                        
                        # Sınıf indeksinin geçerli olup olmadığını kontrol et
                        if 0 <= class_idx < len(class_names):
                            class_counts[class_idx] += 1
                            file_class_counts[class_idx] += 1
                        else:
                            print(f"Uyarı: {file_name} dosyası, satır {line_num}: "
                                  f"Geçersiz sınıf indeksi ({class_idx})")
                    
                    except ValueError:
                        print(f"Uyarı: {file_name} dosyası, satır {line_num}: "
                              f"İlk değer bir sayı değil: '{parts[0]}'")
            
            # Dosya bazlı istatistikleri göster (opsiyonel)
            if file_class_counts:
                print(f"\n{file_name} dosyasındaki sınıf dağılımı:")
                for idx, count in sorted(file_class_counts.items()):
                    print(f"  {idx} ({class_names[idx]}): {count}")
        
        except Exception as e:
            print(f"Hata: {txt_file} dosyası okunurken bir sorun oluştu: {e}")
    
    return class_counts


def main():
    # Argüman ayrıştırıcıyı oluştur
    parser = argparse.ArgumentParser(
        description='Dizindeki txt(yolo formatındaki) dosyalarından sınıf sayılarını hesaplar.'
    )
    parser.add_argument(
        'directory',
        help='Txt dosyalarının bulunduğu dizin yolu'
    )
    parser.add_argument(
        'yaml_file',
        help='Sınıf isimlerini içeren data.yaml dosyasının yolu'
    )
    
    # İsteğe bağlı argüman: çıktı formatı
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'csv', 'json'],
        default='text',
        help='Çıktı formatı (varsayılan: text)'
    )
    
    # Argümanları ayrıştır
    args = parser.parse_args()
    
    # Dizin kontrolü
    if not os.path.isdir(args.directory):
        print(f"Hata: Belirtilen dizin bulunamadı: {args.directory}")
        sys.exit(1)
    
    # YAML dosyasını oku ve sınıf isimlerini al
    class_names = read_yaml_classes(args.yaml_file)
    
    # Sınıf sayılarını hesapla
    class_counts = count_classes_in_directory(args.directory, class_names)
    
    # Sonuçları göster
    print("\n" + "="*50)
    print("SONUÇLAR:")
    print("="*50)
    
    if not class_counts:
        print("Hiçbir sınıf bulunamadı!")
        return
    
    total_objects = sum(class_counts.values())
    print(f"Toplam nesne sayısı: {total_objects}")
    print("-"*30)
    
    # Format'a göre çıktı ver
    if args.format == 'text':
        for idx, count in sorted(class_counts.items()):
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            print(f"Sınıf {idx} ({class_names[idx]}): {count} nesne (%{percentage:.2f})")
    
    elif args.format == 'csv':
        print("indeks,isim,sayi,yuzde")
        for idx, count in sorted(class_counts.items()):
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            print(f"{idx},{class_names[idx]},{count},{percentage:.2f}")
    
    elif args.format == 'json':
        import json
        result = {
            'total_objects': total_objects,
            'classes': []
        }
        for idx, count in sorted(class_counts.items()):
            result['classes'].append({
                'index': idx,
                'name': class_names[idx],
                'count': count,
                'percentage': (count / total_objects) * 100 if total_objects > 0 else 0
            })
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
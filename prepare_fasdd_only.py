
import os
import shutil
import time
from pathlib import Path

# Paths
FASDD_ROOT = r'c:\Users\muham\Desktop\ysa\FASDD_CV\FASDD_CV'
FASDD_IMAGES = os.path.join(FASDD_ROOT, 'images')
FASDD_LABELS = os.path.join(FASDD_ROOT, 'annotations', 'YOLO_CV', 'labels')
FASDD_SPLITS = os.path.join(FASDD_ROOT, 'annotations', 'YOLO_CV')

DEST_ROOT = r'c:\Users\muham\Desktop\ysa\images'
DEST_TRAIN_IMG = os.path.join(DEST_ROOT, 'train', 'images')
DEST_TRAIN_LBL = os.path.join(DEST_ROOT, 'train', 'labels')
DEST_TEST_IMG = os.path.join(DEST_ROOT, 'test', 'images')
DEST_TEST_LBL = os.path.join(DEST_ROOT, 'test', 'labels')

def backup_existing_images():
    if os.path.exists(DEST_ROOT):
        timestamp = int(time.time())
        backup_name = f"{DEST_ROOT}_backup_{timestamp}"
        print(f"Mevcut 'images' klasörü yedekleniyor: {backup_name}")
        try:
            os.rename(DEST_ROOT, backup_name)
        except Exception as e:
            print(f"Yedekleme hatası: {e}")
            print("Lütfen klasörü kullanan uygulamaları kapatın.")
            exit(1)

def process_split(split_file, dest_img_dir, dest_lbl_dir):
    if not os.path.exists(split_file):
        print(f"Uyarı: {split_file} bulunamadı!")
        return

    with open(split_file, 'r') as f:
        lines = f.readlines()
    
    count = 0
    skipped_background = 0
    
    print(f"İşleniyor: {os.path.basename(split_file)} -> Hedef: {os.path.basename(os.path.dirname(dest_img_dir))}")
    total_lines = len(lines)
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # Progress update every 1000 items
        if i % 1000 == 0:
            print(f"  İlerleme: {i}/{total_lines} (%{int(i/total_lines*100)}) - Kopyalanan: {count}")

        filename = os.path.basename(line)
        
        # Sadece Fire ve Smoke resimlerini al (FASDD isimlendirme formatına göre)
        # Background resimleri 'neitherFireNorSmoke' ile başlıyor.
        if filename.startswith('neitherFireNorSmoke'):
            skipped_background += 1
            continue
            
        src_img_path = os.path.join(FASDD_IMAGES, filename)
        
        base_name = os.path.splitext(filename)[0]
        label_name = base_name + '.txt'
        src_lbl_path = os.path.join(FASDD_LABELS, label_name)
        
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, os.path.join(dest_img_dir, filename))
            
            if os.path.exists(src_lbl_path):
                shutil.copy(src_lbl_path, os.path.join(dest_lbl_dir, label_name))
            count += 1

    print(f"  Tamamlandı: {os.path.basename(split_file)}")
    print(f"  Toplam Kopyalanan: {count}")
    print(f"  Atlanan Arka Plan: {skipped_background}")
    print("-" * 30)

def main():
    print("FASDD Hazırlık Süreci Başlıyor...")
    
    # 1. Mevcut klasörü yedekle
    backup_existing_images()
    
    # 2. Yeni yapıyı oluştur
    for d in [DEST_TRAIN_IMG, DEST_TRAIN_LBL, DEST_TEST_IMG, DEST_TEST_LBL]:
        os.makedirs(d, exist_ok=True)
        
    # 3. Kopyalama İşlemleri
    # Train = Train + Val (Daha fazla veri için)
    process_split(os.path.join(FASDD_SPLITS, 'train.txt'), DEST_TRAIN_IMG, DEST_TRAIN_LBL)
    process_split(os.path.join(FASDD_SPLITS, 'val.txt'), DEST_TRAIN_IMG, DEST_TRAIN_LBL)
    
    # Test = Test
    process_split(os.path.join(FASDD_SPLITS, 'test.txt'), DEST_TEST_IMG, DEST_TEST_LBL)
    
    print("İşlem Başarıyla Tamamlandı!")
    print(f"Yeni veri seti: {DEST_ROOT}")

if __name__ == "__main__":
    main()

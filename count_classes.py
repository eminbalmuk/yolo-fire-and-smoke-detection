import os
from collections import defaultdict
from tqdm import tqdm

def count_classes(labels_dir):
    """
    Belirtilen klasördeki YOLO etiket dosyalarını tarar ve sınıf sayılarını çıkarır.
    """
    class_counts = defaultdict(int)
    image_counts = defaultdict(int) # Hangi sınıftan kaç resim var (bir resimde birden fazla aynı sınıf olabilir)
    background_count = 0
    total_files = 0

    # Klasördeki tüm .txt dosyalarını listele
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    total_files = len(label_files)

    print(f"Klasör taranıyor: {labels_dir}")
    print(f"Toplam dosya sayısı: {total_files}")

    for label_file in tqdm(label_files):
        file_path = os.path.join(labels_dir, label_file)
        
        # Dosya boşsa veya sadece boşluk varsa background olarak say
        if os.path.getsize(file_path) == 0:
            background_count += 1
            continue

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                background_count += 1
                continue
                
            # Dosyadaki sınıfları set olarak al (bir resimde 3 yangın varsa 1 yangınlı resim olarak saymak için)
            classes_in_image = set()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    classes_in_image.add(class_id)
            
            for class_id in classes_in_image:
                image_counts[class_id] += 1
                
        except Exception as e:
            print(f"Hata: {file_path} okunamadı. {e}")

    return class_counts, image_counts, background_count, total_files

if __name__ == "__main__":
    # dataset.yaml'a göre train yolu: images/train/images -> demek ki labels: images/train/labels
    # Ancak emin olmak için kullanıcıdan teyit almıştık, images/train/labels mevcut.
    
    train_labels_path = os.path.join("images", "train", "labels")
    val_labels_path = os.path.join("images", "test", "labels")
    
    # Train sayımı
    if os.path.exists(train_labels_path):
        print("\n--- TRAIN Veri Seti ---")
        c_counts, i_counts, bg_count, total = count_classes(train_labels_path)
        print(f"\nSonuçlar:")
        print(f"Toplam Resim: {total}")
        print(f"Arka Plan (Boş) Resim: {bg_count}")
        print(f"Fire (0) Nesne Sayısı: {c_counts[0]}, İçeren Resim Sayısı: {i_counts[0]}")
        print(f"Smoke (1) Nesne Sayısı: {c_counts[1]}, İçeren Resim Sayısı: {i_counts[1]}")
    else:
        print(f"Klasör bulunamadı: {train_labels_path}")

    # Val sayımı
    if os.path.exists(val_labels_path):
        print("\n--- VAL/TEST Veri Seti ---")
        c_counts, i_counts, bg_count, total = count_classes(val_labels_path)
        print(f"\nSonuçlar:")
        print(f"Toplam Resim: {total}")
        print(f"Arka Plan (Boş) Resim: {bg_count}")
        print(f"Fire (0) Nesne Sayısı: {c_counts[0]}, İçeren Resim Sayısı: {i_counts[0]}")
        print(f"Smoke (1) Nesne Sayısı: {c_counts[1]}, İçeren Resim Sayısı: {i_counts[1]}")
    else:
        print(f"Klasör bulunamadı: {val_labels_path}")

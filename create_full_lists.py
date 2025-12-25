
import os

FASDD_ROOT = r'c:\Users\muham\Desktop\ysa\FASDD_CV\FASDD_CV'
ANNOTATIONS_DIR = os.path.join(FASDD_ROOT, 'annotations', 'YOLO_CV')

def create_full_lists():
    print("Tam FASDD listeleri hazırlanıyor (Arka plan resimleri DAHİL)...")
    
    # Dosya yollarını tanımla
    train_orig = os.path.join(ANNOTATIONS_DIR, 'train.txt')
    val_orig = os.path.join(ANNOTATIONS_DIR, 'val.txt')
    test_orig = os.path.join(ANNOTATIONS_DIR, 'test.txt')
    
    train_val_full_path = os.path.join(ANNOTATIONS_DIR, 'train_val_full.txt')
    
    # Train ve Val dosyalarını oku
    with open(train_orig, 'r') as f:
        train_lines = f.readlines()
    
    with open(val_orig, 'r') as f:
        val_lines = f.readlines()
        
    # Test dosyasını oku (orijinalini kullanacağız ama istatistik için)
    with open(test_orig, 'r') as f:
        test_lines = f.readlines()
        
    # Birleştir
    full_train_lines = train_lines + val_lines
    
    # Yeni dosyayı yaz
    with open(train_val_full_path, 'w') as f:
        f.writelines(full_train_lines)
        
    print(f"Oluşturuldu: {train_val_full_path}")
    print(f"  Train Setinden: {len(train_lines)}")
    print(f"  Val Setinden:   {len(val_lines)}")
    print(f"  Toplam Eğitim:  {len(full_train_lines)}")
    print(f"  Test Seti:      {len(test_lines)} (Değişmedi)")
    print("-" * 30)

if __name__ == "__main__":
    create_full_lists()

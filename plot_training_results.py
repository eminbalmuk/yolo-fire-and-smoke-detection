"""
Eğitim Sonuçlarını Seaborn ile Görselleştirme
64 Epoch YOLO Eğitimi için Kayıp ve Metrik Grafikleri
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Seaborn stilini ayarla
sns.set_theme(style="whitegrid")

# Veriyi oku
results_path = Path(__file__).parent / "runs/detect/dfire-yolo/results.csv"
df = pd.read_csv(results_path)

# Sütun isimlerindeki boşlukları temizle
df.columns = df.columns.str.strip()

# Toplam kayıp hesapla (box_loss + cls_loss + dfl_loss)
df['train_total_loss'] = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
df['val_total_loss'] = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']

# Türkçe fontları desteklemek için
plt.rcParams['font.family'] = 'DejaVu Sans'

# Figür oluştur: 2 satır 1 sütun
fig, axes = plt.subplots(2, 1, figsize=(6, 10))

# ============ Şekil 1: Eğitim ve Doğrulama Kaybı ============
ax1 = axes[0]
sns.lineplot(x='epoch', y='train_total_loss', data=df, ax=ax1, color='#1f77b4', linewidth=1.5, label='Eğitim Kaybı')
sns.lineplot(x='epoch', y='val_total_loss', data=df, ax=ax1, color='#ff7f0e', linewidth=1.5, label='Doğrulama Kaybı')

ax1.set_xlabel('Epok Sayısı', fontsize=12)
ax1.set_ylabel('Kayıp', fontsize=12)
ax1.set_title('Eğitim ve Doğrulama Kaybı', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim(0, 64)

# Detaylı ölçekler - Grafik 1
import numpy as np
ax1.set_xticks(np.arange(0, 65, 10))  # Ana x tick'leri
ax1.set_xticks(np.arange(0, 65, 5), minor=True)  # Minor x tick'leri
ax1.set_yticks(np.arange(0, 7, 1))  # Y ekseni 1'er aralıklarla
ax1.tick_params(axis='both', which='major', labelsize=9)
ax1.tick_params(axis='both', which='minor', length=3)
ax1.grid(True, which='major', linestyle='-', alpha=0.7)
ax1.grid(True, which='minor', linestyle=':', alpha=0.4)

# ============ Şekil 2: Eğitim ve Doğrulama Başarısı (mAP) ============
ax2 = axes[1]
# Eğitim başarısı olarak precision kullanıyoruz
# Doğrulama başarısı olarak mAP50 kullanıyoruz
sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=ax2, color='#1f77b4', linewidth=1.5, label='Eğitim Başarısı (Precision)')
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=ax2, color='#ff7f0e', linewidth=1.5, label='Doğrulama Başarısı (mAP50)')

ax2.set_xlabel('Epok Sayısı', fontsize=12)
ax2.set_ylabel('Başarı', fontsize=12)
ax2.set_title('Eğitim ve Doğrulama Başarısı', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xlim(0, 64)
ax2.set_ylim(0, 1)

# Detaylı ölçekler - Grafik 2
ax2.set_xticks(np.arange(0, 65, 10))  # Ana x tick'leri
ax2.set_xticks(np.arange(0, 65, 5), minor=True)  # Minor x tick'leri
ax2.set_yticks(np.arange(0, 1.1, 0.1))  # Y ekseni 0.1 aralıklarla
ax2.tick_params(axis='both', which='major', labelsize=9)
ax2.tick_params(axis='both', which='minor', length=3)
ax2.grid(True, which='major', linestyle='-', alpha=0.7)
ax2.grid(True, which='minor', linestyle=':', alpha=0.4)

# Layout'u düzenle
plt.tight_layout()

# Grafikleri kaydet
output_path = Path(__file__).parent / "runs/detect/dfire-yolo/egitim_grafikleri.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Grafikler kaydedildi: {output_path}")

# Göster
plt.show()

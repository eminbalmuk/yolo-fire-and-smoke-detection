
import os
import shutil
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

# Ensure destination directories exist
for d in [DEST_TRAIN_IMG, DEST_TRAIN_LBL, DEST_TEST_IMG, DEST_TEST_LBL]:
    os.makedirs(d, exist_ok=True)

def process_split(split_file, dest_img_dir, dest_lbl_dir):
    with open(split_file, 'r') as f:
        lines = f.readlines()
    
    count = 0
    skipped_background = 0
    missing = 0
    
    print(f"Processing {split_file}...")
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Extract filename from './images/filename.jpg'
        filename = os.path.basename(line)
        
        # Filter: Exclude background images (neitherFireNorSmoke)
        if filename.startswith('neitherFireNorSmoke'):
            skipped_background += 1
            continue
            
        src_img_path = os.path.join(FASDD_IMAGES, filename)
        
        # Label file logic: same name, .txt extension
        base_name = os.path.splitext(filename)[0]
        label_name = base_name + '.txt'
        src_lbl_path = os.path.join(FASDD_LABELS, label_name)
        
        if os.path.exists(src_img_path):
            # Copy Image
            shutil.copy(src_img_path, os.path.join(dest_img_dir, filename))
            
            # Copy Label
            if os.path.exists(src_lbl_path):
                shutil.copy(src_lbl_path, os.path.join(dest_lbl_dir, label_name))
            else:
                # If label missing for a fire/smoke image, that's an issue, but maybe create empty?
                # Usually valid YOLO data has labels. 
                # If it's fire/smoke, it should have a label. 
                # If no label file, YOLO considers it background. 
                # But we filtered based on name.
                pass
            count += 1
        else:
            missing += 1
            # print(f"Missing: {src_img_path}")

    print(f"Processed {len(lines)} lines.")
    print(f"Copied: {count}")
    print(f"Skipped Background: {skipped_background}")
    print(f"Missing Sources: {missing}")
    print("-" * 30)

# Process Train -> Train
process_split(os.path.join(FASDD_SPLITS, 'train.txt'), DEST_TRAIN_IMG, DEST_TRAIN_LBL)

# Process Val -> Train (User wants to use it for training)
process_split(os.path.join(FASDD_SPLITS, 'val.txt'), DEST_TRAIN_IMG, DEST_TRAIN_LBL)

# Process Test -> Test
process_split(os.path.join(FASDD_SPLITS, 'test.txt'), DEST_TEST_IMG, DEST_TEST_LBL)

print("Merge completed.")

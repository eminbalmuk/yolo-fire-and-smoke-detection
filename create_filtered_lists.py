
import os

# Paths
FASDD_ROOT = r'c:\Users\muham\Desktop\ysa\FASDD_CV\FASDD_CV'
ANNOTATIONS_DIR = os.path.join(FASDD_ROOT, 'annotations', 'YOLO_CV')

def filter_list(input_filename, output_filename):
    input_path = os.path.join(ANNOTATIONS_DIR, input_filename)
    output_path = os.path.join(ANNOTATIONS_DIR, output_filename)
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
        
    filtered_lines = []
    skipped = 0
    
    for line in lines:
        # Check if line contains 'neitherFireNorSmoke' (background class)
        if 'neitherFireNorSmoke' in line:
            skipped += 1
            continue
        filtered_lines.append(line)
        
    with open(output_path, 'w') as f:
        f.writelines(filtered_lines)
        
    print(f"Processed {input_filename} -> {output_filename}")
    print(f"  Total: {len(lines)}")
    print(f"  Kept: {len(filtered_lines)}")
    print(f"  Skipped (Background): {skipped}")
    print("-" * 30)
    
    # Return path relative to project root for dataset.yaml if needed, 
    # but dataset.yaml will use absolute paths or relative to its own location.

# Create filtered lists
# We will combine train and val for training as per user preference
filter_list('train.txt', 'train_filtered.txt')
filter_list('val.txt', 'val_filtered.txt') # We'll create it anyway
filter_list('test.txt', 'test_filtered.txt')

# Create a combined train set (train + val)
with open(os.path.join(ANNOTATIONS_DIR, 'train_filtered.txt'), 'r') as f1, \
     open(os.path.join(ANNOTATIONS_DIR, 'val_filtered.txt'), 'r') as f2:
    combined = f1.readlines() + f2.readlines()

combined_path = os.path.join(ANNOTATIONS_DIR, 'train_val_filtered.txt')
with open(combined_path, 'w') as f:
    f.writelines(combined)

print(f"Combined Train+Val created at: {combined_path}")
print(f"Total training samples: {len(combined)}")

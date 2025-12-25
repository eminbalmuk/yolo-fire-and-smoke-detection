
import os

FASDD_ROOT = r'c:\Users\muham\Desktop\ysa\FASDD_CV\FASDD_CV'
ANNOTATIONS_DIR = os.path.join(FASDD_ROOT, 'annotations', 'YOLO_CV')

def convert_to_absolute(input_filename, output_filename):
    input_path = os.path.join(ANNOTATIONS_DIR, input_filename)
    output_path = os.path.join(ANNOTATIONS_DIR, output_filename)
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
        
    abs_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Remove ./ prefix if present
        if line.startswith('./'):
            clean_path = line[2:]
        else:
            clean_path = line
            
        # Join with root
        abs_path = os.path.join(FASDD_ROOT, clean_path)
        abs_lines.append(abs_path + '\n')
        
    with open(output_path, 'w') as f:
        f.writelines(abs_lines)
        
    print(f"Converted {input_filename} -> {output_filename} (Absolute Paths)")
    print(f"Sample line: {abs_lines[0].strip()}")

# Convert the full training list
convert_to_absolute('train_val_full.txt', 'train_val_full_abs.txt')
# Convert test list
convert_to_absolute('test.txt', 'test_abs.txt')

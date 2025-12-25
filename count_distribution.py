
import os
from collections import Counter

base_dir = r'c:\Users\muham\Desktop\ysa\FASDD_CV\FASDD_CV\annotations\YOLO_CV'
files = ['train.txt', 'val.txt', 'test.txt']

all_lines = []
for fname in files:
    p = os.path.join(base_dir, fname)
    if os.path.exists(p):
        with open(p, 'r') as f:
            all_lines.extend([l.strip() for l in f.readlines()])

unique_lines = list(set(all_lines))
print(f"Total unique lines: {len(unique_lines)}")

counts = Counter()
for line in unique_lines:
    if 'neitherFireNorSmoke' in line:
        counts['empty'] += 1
    elif 'bothFireAndSmoke' in line:
        counts['both'] += 1
    elif 'fire_' in line:
        counts['fire'] += 1
    elif 'smoke_' in line:
        counts['smoke'] += 1
    else:
        counts['unknown'] += 1

print("Counts:", counts)

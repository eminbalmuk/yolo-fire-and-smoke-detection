
import os
import random

# Seed for reproducibility
random.seed(42)

# Paths
FASDD_ROOT = r'c:\Users\muham\Desktop\ysa\FASDD_CV\FASDD_CV'
ANNOTATIONS_DIR = os.path.join(FASDD_ROOT, 'annotations', 'YOLO_CV')
FILES = ['train.txt', 'val.txt', 'test.txt']

def to_absolute(line):
    line = line.strip()
    if line.startswith('./'):
        line = line[2:]
    return os.path.join(FASDD_ROOT, line)

def get_filename(path):
    return os.path.basename(path)

def prepare_dataset():
    print("Loading file lists...")
    all_lines = []
    for fname in FILES:
        path = os.path.join(ANNOTATIONS_DIR, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = [to_absolute(l) for l in f.readlines() if l.strip()]
                all_lines.extend(lines)
    
    # Deduplicate
    all_lines = list(set(all_lines))
    print(f"Total unique images found: {len(all_lines)}")

    # Classify
    fire_imgs = []
    smoke_imgs = []
    empty_imgs = []
    both_imgs = []
    others = []

    for img in all_lines:
        fname = get_filename(img)
        if 'neitherFireNorSmoke' in fname:
            empty_imgs.append(img)
        elif 'bothFireAndSmoke' in fname:
            both_imgs.append(img)
        elif 'fire_' in fname:
            fire_imgs.append(img)
        elif 'smoke_' in fname:
            smoke_imgs.append(img)
        else:
            others.append(img)

    print(f"Counts found:")
    print(f"  Fire: {len(fire_imgs)}")
    print(f"  Smoke: {len(smoke_imgs)}")
    print(f"  Empty: {len(empty_imgs)}")
    print(f"  Both: {len(both_imgs)}")
    print(f"  Others: {len(others)}")

    # Shuffle everything
    random.shuffle(fire_imgs)
    random.shuffle(smoke_imgs)
    random.shuffle(empty_imgs)
    random.shuffle(both_imgs)

    # Training Set Requirement: 6000 Fire, 6000 Smoke, 6000 Empty
    train_fire = fire_imgs[:6000]
    train_smoke = smoke_imgs[:6000]
    train_empty = empty_imgs[:6000]

    # Remaining for Test
    rem_fire = fire_imgs[6000:]
    rem_smoke = smoke_imgs[6000:]
    rem_empty = empty_imgs[6000:]
    rem_both = both_imgs # We didn't use both for training

    # Test Set Requirement: 4000 Total
    # Let's distribute evenly-ish: 1000 Fire, 1000 Smoke, 1000 Empty, 1000 Both
    # Check availability
    n_test_fire = min(1000, len(rem_fire))
    n_test_smoke = min(1000, len(rem_smoke))
    n_test_empty = min(1000, len(rem_empty))
    n_test_both = min(1000, len(rem_both))

    test_fire = rem_fire[:n_test_fire]
    test_smoke = rem_smoke[:n_test_smoke]
    test_empty = rem_empty[:n_test_empty]
    test_both = rem_both[:n_test_both]

    # If we need more to reach 4000, fill from remaining of any class
    current_test_count = len(test_fire) + len(test_smoke) + len(test_empty) + len(test_both)
    needed = 4000 - current_test_count
    
    if needed > 0:
        pool = (rem_fire[n_test_fire:] + rem_smoke[n_test_smoke:] + 
                rem_empty[n_test_empty:] + rem_both[n_test_both:])
        random.shuffle(pool)
        extras = pool[:needed]
        # Just add to one of the lists or keep separate, doesn't matter for the final file
        test_fire.extend(extras) # extending blindly, but it's just a list for writing
    
    # Construct final lists
    train_set = train_fire + train_smoke + train_empty
    test_set = test_fire + test_smoke + test_empty + test_both
    
    random.shuffle(train_set)
    random.shuffle(test_set)

    print(f"Selected Training Set: {len(train_set)}")
    print(f"  Fire: {len(train_fire)}")
    print(f"  Smoke: {len(train_smoke)}")
    print(f"  Empty: {len(train_empty)}")
    
    print(f"Selected Test Set: {len(test_set)}")

    # Write files
    train_out = os.path.join(ANNOTATIONS_DIR, 'train_balanced_18k_abs.txt')
    test_out = os.path.join(ANNOTATIONS_DIR, 'test_balanced_4k_abs.txt')

    with open(train_out, 'w') as f:
        f.writelines([l + '\n' for l in train_set])
        
    with open(test_out, 'w') as f:
        f.writelines([l + '\n' for l in test_set])

    print(f"Written train list to: {train_out}")
    print(f"Written test list to: {test_out}")

    # Create YAML
    yaml_path = os.path.join(r'c:\Users\muham\Desktop\ysa', 'dataset_balanced.yaml')
    yaml_content = f"""
# Auto-generated balanced dataset config
path: .
train: {train_out}
val: {test_out}
test: {test_out}

names:
  0: fire
  1: smoke
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset config at: {yaml_path}")

if __name__ == '__main__':
    prepare_dataset()

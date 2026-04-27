import os
import shutil

def cleanup_dataset(base_path):
    """Removes __MACOSX and .DS_Store files."""
    for root, dirs, files in os.walk(base_path, topdown=False):
        for name in dirs:
            if name == '__MACOSX':
                shutil.rmtree(os.path.join(root, name))
                print(f"Removed: {os.path.join(root, name)}")
        for name in files:
            if name == '.DS_Store' or name.startswith('._'):
                os.remove(os.path.join(root, name))
                print(f"Removed: {os.path.join(root, name)}")

def summarize_dataset(base_path):
    """Prints a summary of the dataset structure."""
    print("\nDataset Summary:")
    print("-" * 20)
    
    # The zip structure was chest_xray/chest_xray/...
    # We extracted to 'dataset', so it might be dataset/chest_xray/chest_xray/...
    # Let's find the actual root
    actual_root = base_path
    for _ in range(3): # Look up to 3 levels deep for 'train'
        if os.path.exists(os.path.join(actual_root, 'train')):
            break
        subdirs = [d for d in os.listdir(actual_root) if os.path.isdir(os.path.join(actual_root, d))]
        if subdirs:
            actual_root = os.path.join(actual_root, subdirs[0])
        else:
            break
            
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(actual_root, split)
        if os.path.exists(split_path):
            print(f"Split: {split}")
            for label in os.listdir(split_path):
                label_path = os.path.join(split_path, label)
                if os.path.isdir(label_path):
                    count = len([f for f in os.listdir(label_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                    print(f"  - {label}: {count} images")
        else:
            print(f"Split: {split} NOT FOUND at {split_path}")

if __name__ == "__main__":
    dataset_path = "dataset"
    if os.path.exists(dataset_path):
        cleanup_dataset(dataset_path)
        summarize_dataset(dataset_path)
    else:
        print("Dataset directory not found.")

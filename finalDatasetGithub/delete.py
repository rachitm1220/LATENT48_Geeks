import os
import random
import shutil

def keep_random_images(folder_path, keep_count=5):
    """Keep a random sample of images in a folder, delete the rest."""
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.ico')
    files = [f for f in os.listdir(folder_path) 
             if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(folder_path, f))]
    
    if len(files) <= keep_count:
        print(f"Only {len(files)} images found. Nothing to delete.")
        return
    
    # Select random images to keep
    to_keep = random.sample(files, keep_count)
    to_delete = [f for f in files if f not in to_keep]
    
    # Delete files
    deleted_count = 0
    for filename in to_delete:
        filepath = os.path.join(folder_path, filename)
        try:
            os.remove(filepath)
            print(f"Deleted: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete {filename}: {e}")
    
    print(f"\nDone! Kept {len(to_keep)} images, deleted {deleted_count} images.")


# Usage example
if __name__ == "__main__":
    folder = "objectsDetection/objectsDetected"  # Change this to your folder path
    keep_random_images(folder, keep_count=5)
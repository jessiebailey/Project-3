import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to your main folder
main_folder = "/Users/arianaelahi/Desktop/realwaste-main/RealWaste"

# 1. Count total images and folders
folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
total_images = 0
image_sizes = []
image_formats = []
folder_counts = {}

print("Starting analysis...\n")

for folder in folders:
    folder_path = os.path.join(main_folder, folder)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    count = len(images)
    folder_counts[folder] = count
    total_images += count
    
    # Get size and format of each image
    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        try:
            with Image.open(img_path) as img:
                image_sizes.append(img.size)  # (width, height)
                image_formats.append(img.format)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

# 2. Basic stats
print(f"Total folders: {len(folders)}")
print(f"Total images: {total_images}")
print(f"Image formats found: {set(image_formats)}")

# 3. Images per folder
print("\nImages per folder:")
for folder, count in sorted(folder_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {folder}: {count} images")

# 4. Image dimensions summary
widths = [size[0] for size in image_sizes]
heights = [size[1] for size in image_sizes]

print(f"\nImage dimensions:")
print(f"  Width  - min: {min(widths)}, max: {max(widths)}, avg: {sum(widths)/len(widths):.1f}")
print(f"  Height - min: {min(heights)}, max: {max(heights)}, avg: {sum(heights)/len(heights):.1f}")

# 5. Plot: Images per folder
plt.figure(figsize=(10, 5))
plt.bar(folder_counts.keys(), folder_counts.values())
plt.title("Number of Images per Folder")
plt.xlabel("Folder")
plt.ylabel("Image Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Plot: Image size distribution
plt.figure(figsize=(8, 5))
plt.hist(widths, bins=30, alpha=0.7, label='Width')
plt.hist(heights, bins=30, alpha=0.7, label='Height')
plt.title("Distribution of Image Widths and Heights")
plt.xlabel("Pixels")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

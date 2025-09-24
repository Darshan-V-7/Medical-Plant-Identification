import os
from PIL import Image
from rembg import remove

# Source and destination directories
SOURCE_DIR = r'C:\Users\Darshan\OneDrive\Desktop\Medical Plant Identification\plant_dataset'
DEST_DIR = 'cleaned_leaf_dataset'

# Ensure destination structure is in place
os.makedirs(DEST_DIR, exist_ok=True)

def remove_bg_and_add_background(input_path, output_path, bg_color=(240, 240, 240)):
    try:
        img = Image.open(input_path).convert("RGBA")
        img_no_bg = remove(img)

        # Create background and paste
        background = Image.new("RGBA", img_no_bg.size, bg_color + (255,))
        result = Image.alpha_composite(background, img_no_bg)

        # Convert to RGB (remove alpha) and save as PNG
        result.convert("RGB").save(output_path.replace(".jpg", ".png"))
        print(f"✅ Processed: {output_path}")
    except Exception as e:
        print(f"❌ Failed: {input_path} — {e}")

def process_dataset():
    for class_folder in os.listdir(SOURCE_DIR):
        class_src_path = os.path.join(SOURCE_DIR, class_folder)
        class_dest_path = os.path.join(DEST_DIR, class_folder)

        if not os.path.isdir(class_src_path):
            continue

        os.makedirs(class_dest_path, exist_ok=True)

        for filename in os.listdir(class_src_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            input_path = os.path.join(class_src_path, filename)
            output_path = os.path.join(class_dest_path, filename)
            remove_bg_and_add_background(input_path, output_path)

if __name__ == "__main__":
    process_dataset()

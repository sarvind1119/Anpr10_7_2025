import os
from vision_utils import call_llama_vision

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:4b"  # Use your actual vision model's tag here!
PROMPT = (
    "Identify the type of this vehicle (SUV, Sedan, Truck, etc.),color of the vehicle, company if available and extract the registration (number plate) number very carefully(Don't skip any character from it). "
    "Respond ONLY in this format (no extra text):\n"
    "Car Type: <type>\n"
    "Color: <color>\n"
    "Company: <company>\n"
    "Registration Number: <number or 'Not Visible'>"
)
INPUT_FOLDER = r"E:\\July2025\\Anpr10_7\\detected_cars\\2025-07-10"
OUTPUT_FOLDER = r"E:\\July2025\\Anpr10_7\\output_descriptions"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- MAIN SCRIPT ---
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(image_extensions):
        img_path = os.path.join(INPUT_FOLDER, filename)
        output_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.txt")
        # Skip if description already exists (optional)
        if os.path.exists(output_file):
            print(f"Skipping {filename} (already described)")
            continue

        try:
            with open(img_path, "rb") as img_file:
                image_bytes = img_file.read()
            print(f"Processing {filename} ...")
            description = call_llama_vision(
                image_bytes,
                PROMPT,
                model=MODEL,
                ollama_url=OLLAMA_URL
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(description)
            print(f"Saved description to {output_file}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print("✅ All images processed.")

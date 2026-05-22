import os
from vision_utils import call_llama_vision

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434"
#MODEL = "gemma3:4b"  # Use your actual vision model's tag here!
MODEL= "qwen2.5vl:7b"
PROMPT = (
    "Identify the type of this vehicle (SUV, Sedan, Truck, etc.),color of the vehicle, company if available and extract the registration (number plate) number very carefully(Don't skip any character from it). "
    "Respond ONLY in this format (no extra text):\n"
    "Car Type: <type>\n"
    "Color: <color>\n"
    "Company: <company>\n"
    "Registration Number: <number or 'Not Visible'>"
)
# PROMPT = (
#     "You are analyzing images of Indian vehicles. "
#     "Your tasks:\n"
#     "1. Identify the type of vehicle (SUV, Sedan, Truck, etc.).\n"
#     "2. Identify the vehicle's color.\n"
#     "3. Identify the company/manufacturer (if visible).\n"
#     "4. Extract the registration (number plate) number CAREFULLY and COMPLETELY.\n\n"
#     "VERY IMPORTANT: Indian vehicle registration numbers follow this pattern:\n"
#     "- First two letters: State code (e.g., MP for Madhya Pradesh, not NP; UP for Uttar Pradesh, etc.).\n"
#     "- Next two digits: District code.\n"
#     "- Next one or two letters: Unique code.\n"
#     "- Last four digits: Vehicle number.\n"
#     "Example: MP09AB1234, UP32GH5678, DL3CAF0987\n"
#     "There are no Indian state codes starting with N. If you see something that looks like 'NP', consider 'MP' as correct.\n\n"
#     "If the registration is unclear, do your best to guess according to this Indian format, but do not hallucinate extra information. "
#     "If the number plate is not visible, state 'Not Visible'.\n"
#     "Respond ONLY in this format (no extra text):\n"
#     "Car Type: <type>\n"
#     "Color: <color>\n"
#     "Company: <company>\n"
#     "Registration Number: <number or 'Not Visible'>"
# )

# PROMPT = (
#     "Analyze this image of an Indian vehicle and answer the following, only based on what is clearly visible in the image:\n"
#     "1. Car Type (e.g., SUV, Sedan, Truck, etc.)\n"
#     "2. Color of the vehicle\n"
#     "3. Company or manufacturer (if visible)\n"
#     "4. Registration Number: Write the registration (number plate) exactly as seen, even if some characters are unclear. "
#     "If the plate is not visible or not readable, write 'Not Visible'. Do not guess or fill missing characters. "
#     "Do not substitute or assume state codes; only write what you see.\n\n"
#     "Format your answer like this (no extra text):\n"
#     "Car Type: <type>\n"
#     "Color: <color>\n"
#     "Company: <company>\n"
#     "Registration Number: <number as seen or 'Not Visible'>"
# )


INPUT_FOLDER = r"E:\\July2025\\Anpr10_7\\detected_cars\\2025-07-21"
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

# utils/vision_utils.py

import requests
import base64
import io
from PIL import Image
#from logger import get_logger

#logger = get_logger(__name__)

OLLAMA_TIMEOUT = 180  # seconds

def call_llama_vision(image_bytes, prompt, model="llama3.2-vision", ollama_url="http://localhost:11434"):
    """
    Send an image and prompt to Ollama vision-capable model.

    Args:
        image_bytes (bytes): Raw image file bytes
        prompt (str): Question related to the image
        model (str): Ollama model name (e.g., llama3.2-vision)
        ollama_url (str): Base URL of Ollama server

    Returns:
        str: Model response or error message
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((512, 512))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_img]
                }
            ],
            "stream": False
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{ollama_url}/api/chat", json=payload, headers=headers, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()

        response_data = response.json()
        return response_data.get("message", {}).get("content", "[No content returned]")

    except requests.exceptions.RequestException as e:
        #logger.error(f"Vision query failed: {e}")
        return f"❌ Vision Request Failed: {e}"

    except Exception as e:
        #logger.error(f"Unexpected error in vision function: {e}")
        return f"❌ Unexpected Vision Error: {e}"

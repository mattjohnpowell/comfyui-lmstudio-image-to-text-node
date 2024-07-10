import base64
import requests
import json
import numpy as np
from PIL import Image
import io

class ExpoLmstudioImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "Describe this image in detail"}),
                "model": ("STRING", {"default": "moondream2-text-model-f16.gguf"}),
                "system_prompt": ("STRING", {"default": "This is a chat between a user and an assistant. The assistant is an expert in describing images, with detail and accuracy"}),
                "ip_address": ("STRING", {"default": "localhost"}),
                "port": ("INT", {"default": 1234, "min": 1, "max": 65535}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)
    FUNCTION = "process_image"
    CATEGORY = "ComfyExpo/I2T"

    def process_image(self, image, user_prompt, model, system_prompt, ip_address, port, debug=False):
        if debug:
            print(f"Debug: Starting process_image method")
            print(f"Debug: Text input: {user_prompt}")
            print(f"Debug: Model: {model}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: IP Address: {ip_address}")
            print(f"Debug: Port: {port}")
            print(f"Debug: Image shape: {image.shape}")

        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(np.uint8(image[0]*255))
            
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            if debug:
                print(f"Debug: Image successfully converted to base64")

            # Prepare the payload for the server
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                "max_tokens": 1000,
                "stream": False
            }

            if debug:
                print(f"Debug: Payload prepared, attempting to connect to server")

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer lm-studio"
            }

            url = f"http://{ip_address}:{port}/v1/chat/completions"
            response = requests.post(url, 
                                     json=payload, 
                                     headers=headers, 
                                     timeout=60)
            
            if debug:
                print(f"Debug: Server response status code: {response.status_code}")

            response.raise_for_status()  # Raise an exception for bad status codes

            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                response_text = response_json['choices'][0]['message']['content']
            else:
                response_text = "No content in the response"

            if debug:
                print(f"Debug: Response received: {response_text[:100]}...")  # Print first 100 characters

            return (response_text,)

        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with the server: {str(e)}"
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            print(error_message)
            return (error_message,)
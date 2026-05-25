"""
@author: Matt John Powell
@title: LM Studio Nodes for ComfyUI
@nickname: LM Studio Nodes
@description: This extension provides three custom nodes for ComfyUI that integrate LM Studio's capabilities:
1. Image to Text: Generates text descriptions of images using vision models.
2. Text Generation: Generates text based on a given prompt using language models.
3. Unified Node: A versatile node that can handle both image and text inputs.
All nodes leverage the LM Studio Python SDK for better performance and reliability.
Includes a fallback mechanism to use any loaded model if the specified model fails.
"""

import base64
import numpy as np
from PIL import Image
import time
import os
import io
import tempfile
import hashlib
import random
import concurrent.futures
import re

_ENV_API_TOKEN = "LM_API_TOKEN"
api_token = os.environ.get(_ENV_API_TOKEN, "lm-studio")

try:
    import lmstudio as lms
except Exception:
    # keep name available for runtime checks; nodes will handle missing SDK at call time
    lms = None

# Default models to use
DEFAULT_LLM = "gemma-3-4b-it-qat"
DEFAULT_VISION = "qwen/qwen3-vl-8b"

# Try to import LM Studio SDK
# lmstudio imported above in a try/except; keep lms as None when unavailable

# No longer checking SDK compatibility

# --- Helper function to strip <think>...</think> blocks from model responses ---
def strip_thinking_tags(text):
    """
    Remove <think>...</think> blocks produced by reasoning/thinking-mode models
    (e.g. Qwen3, DeepSeek-R1, and abliterated variants).  The tags are matched
    case-insensitively and across multiple lines so that the full chain-of-thought
    section is removed regardless of how the model formats it.
    Returns the cleaned text with leading/trailing whitespace stripped.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# --- Helper function to get model info with fallback ---
def get_model_info_with_fallback(model_key, debug=False):
    """
    Attempts to get the model information for use with the LM Studio Python SDK.
    Returns the model key to use or raises an Exception when no model can be obtained.
    """
    if lms is None:
        raise Exception("LM Studio SDK (lmstudio) not available")

    # If model_key is provided and not empty, use it directly
    if model_key and str(model_key).strip() != "":
        if debug:
            print(f"Debug: Using provided model key: '{model_key}'")
        return model_key

    # Try to find a fallback model
    try:
        # Try to get loaded models
        try:
            with lms.Client() as client:
                if hasattr(client, "list_loaded_models"):
                    loaded_models = client.list_loaded_models()
                elif hasattr(client.llm, "list_loaded"):
                    loaded_models = client.llm.list_loaded()
                elif hasattr(client.llm, "list_loaded_models"):
                    loaded_models = client.llm.list_loaded_models()
        except Exception as e:
            if debug:
                print(f"Debug: Failed to get loaded models: {e}")
            loaded_models = None

        if not loaded_models:
            if debug:
                print("Debug: No loaded models found, will use default model")
            return None  # Let the client use default

        # If debugging, show the raw loaded_models structure
        if debug:
            try:
                import pprint
                print("Debug: Raw loaded_models:")
                pprint.pprint(loaded_models)
            except Exception:
                print(f"Debug: loaded_models (repr): {repr(loaded_models)}")

        # Try to extract a usable model key from various shapes
        def _extract_model_name(obj):
            # strings
            if isinstance(obj, str):
                return obj
            # list/tuple -> inspect first element
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                return _extract_model_name(obj[0])
            # dict -> try common keys
            if isinstance(obj, dict):
                for k in ("model", "id", "name"):
                    if k in obj and isinstance(obj[k], str) and obj[k]:
                        return obj[k]
                keys = list(obj.keys())
                if keys and isinstance(keys[0], str):
                    return keys[0]
            # object with attributes
            for attr in ("model", "id", "name", "display_name", "identifier", "model_key"):
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    if isinstance(val, str) and val:
                        return val
            return None

        fallback_model_key = _extract_model_name(loaded_models)
        if fallback_model_key:
            if debug:
                print(f"Debug: Found loaded model '{fallback_model_key}' as fallback.")
            return fallback_model_key
        else:
            if debug:
                print("Debug: Could not extract model name, using default")
            return None  # Let the client use default

    except Exception as e:
        if debug:
            print(f"Debug: Exception in fallback detection: {e}")
        return None  # Let the client use default


def check_lmstudio_connection():
    """
    Verify that LM Studio is reachable before attempting generation.
    Raises a clear exception if the server is not running so ComfyUI halts
    the pipeline rather than passing an error string to downstream nodes.
    """
    if lms is None:
        raise Exception(
            "LM Studio SDK (lmstudio) is not installed. "
            "Run: pip install lmstudio"
        )
    try:
        with lms.Client() as client:  # noqa: F841  – connection test only
            pass
    except Exception as e:
        raise Exception(
            f"Cannot connect to LM Studio. "
            f"Please make sure LM Studio is open and the local server is enabled. "
            f"(Error: {e})"
        ) from e


def safe_get_stats_info(result, debug=False):
    """
    Safely extract statistics information from the result object.
    Handles different SDK versions and attribute names.
    """
    stats_info = {}
    
    if hasattr(result, 'stats') and result.stats:
        # Get predicted tokens count
        if hasattr(result.stats, 'predicted_tokens_count'):
            stats_info['predicted_tokens'] = result.stats.predicted_tokens_count
        elif hasattr(result.stats, 'tokens_count'):
            stats_info['predicted_tokens'] = result.stats.tokens_count
        else:
            stats_info['predicted_tokens'] = "N/A"
        
        # Get time to first token
        if hasattr(result.stats, 'time_to_first_token_sec'):
            stats_info['time_to_first_token'] = result.stats.time_to_first_token_sec
        elif hasattr(result.stats, 'generation_time_sec'):
            stats_info['time_to_first_token'] = result.stats.generation_time_sec
        elif hasattr(result.stats, 'time_to_first_token'):
            stats_info['time_to_first_token'] = result.stats.time_to_first_token
        else:
            stats_info['time_to_first_token'] = "N/A"
        
        # Get stop reason
        if hasattr(result.stats, 'stop_reason'):
            stats_info['stop_reason'] = result.stats.stop_reason
        else:
            stats_info['stop_reason'] = "N/A"
    else:
        stats_info = {
            'predicted_tokens': "N/A",
            'time_to_first_token': "N/A",
            'stop_reason': "N/A"
        }
    
    if debug:
        print(f"Debug: Stats extraction - Tokens: {stats_info['predicted_tokens']}, Time: {stats_info['time_to_first_token']}, Stop reason: {stats_info['stop_reason']}")
    
    return stats_info


class ExpoLmstudioUnified:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"default": "give me a prompt for an image generation"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "model_key": ("STRING", {"default": DEFAULT_LLM}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "process_input"
    CATEGORY = "ComfyExpo/LMStudio"

    @classmethod
    def IS_CHANGED(cls, text_input, system_prompt, model_key, auto_unload, unload_delay, seed, image=None, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300):
        m = hashlib.sha256()
        
        m.update(str(text_input).encode())
        m.update(str(system_prompt).encode())
        m.update(str(model_key).encode())
        m.update(str(auto_unload).encode())
        m.update(str(unload_delay).encode())
        m.update(str(seed).encode())
        m.update(str(max_tokens).encode())
        m.update(str(temperature).encode())
        m.update(str(debug).encode())
        m.update(str(timeout_seconds).encode())
        
        # Include image hash if present
        if image is not None:
            # Convert image to a hashable representation
            image_bytes = np.array(image).tobytes()
            m.update(image_bytes)
        
        return m.hexdigest()

    def process_input(self, text_input, system_prompt, model_key, auto_unload, unload_delay, seed, image=None, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300):
        # Normalize debug: accept both bool (BOOLEAN widget) and string (legacy/fallback)
        debug = debug if isinstance(debug, bool) else str(debug).lower() == "true"
        # Fail fast if LM Studio is not reachable
        check_lmstudio_connection()

        # Check if we have valid inputs
        has_image = image is not None
        has_text = text_input.strip() != ""

        # If no inputs are provided, return a message
        if not has_image and not has_text:
            return ("No inputs provided. Please connect an image or provide text input.",)

        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting unified process_input method")
            print(f"Debug: Has image input: {has_image}")
            print(f"Debug: Has text input: {has_text}")
            print(f"Debug: Requested Model: {model_key}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")

        managed_temp_path = [None]
        timed_out = [False]

        def do_the_work():
            with lms.Client() as client:
                # --- Get model info and create client context ---
                model_key_to_use = get_model_info_with_fallback(model_key, debug)

                # Get model with proper context management
                if model_key_to_use:
                    if auto_unload == "True" and unload_delay > 0:
                        model = client.llm.model(model_key_to_use, ttl=unload_delay)
                    else:
                        model = client.llm.model(model_key_to_use)
                else:
                    # Use default model
                    model = client.llm.model()

                chat = lms.Chat(system_prompt)

                # Process inputs
                if has_image:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(np.uint8(image[0]*255))

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        managed_temp_path[0] = temp_file.name
                        # Save to the temporary file
                        pil_image.save(managed_temp_path[0], format="JPEG")

                    if debug:
                        print(f"Debug: Saved image to temporary file: {managed_temp_path[0]}")

                    # Use the client's files namespace to prepare the image
                    image_handle = client.files.prepare_image(managed_temp_path[0])

                    # Add user message with correct signature per SDK docs
                    if has_text:
                        chat.add_user_message(text_input, images=[image_handle])
                        if debug:
                            print(f"Debug: Added text and image to chat message")
                    else:
                        chat.add_user_message(images=[image_handle])
                        if debug:
                            print(f"Debug: Added image only to chat message")
                elif has_text:
                    # Add user message with text only
                    chat.add_user_message(text_input)
                    if debug:
                        print(f"Debug: Added text only to chat message")

                # Configure generation parameters
                config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "seed": seed
                }

                if debug:
                    print(f"Debug: Sending request to LM Studio with config: {config}")

                result = model.respond(chat, config=config)

                output_text = strip_thinking_tags(result.content)

                if debug:
                    print(f"Debug: Response received: {output_text[:100]}...")  # Print first 100 characters

                # Extract and log stats information
                stats_info = safe_get_stats_info(result, debug)
                if debug:
                    print(f"Debug: Tokens generated: {stats_info['predicted_tokens']}, Time to first token: {stats_info['time_to_first_token']}s")

                # Unload model immediately if requested (TTL handles delayed unloading)
                if auto_unload == "True" and unload_delay == 0:
                    try:
                        if debug:
                            print(f"Debug: Unloading model immediately.")
                        model.unload()
                    except Exception as unload_err:
                        print(f"Warning: Failed to unload model: {unload_err}")

                return (output_text,)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(do_the_work)
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            timed_out[0] = True
            error_message = f"Error: LM Studio operation timed out after {timeout_seconds} seconds. The connection may be unstable."
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"LM Studio error (Unified node): {str(e)}"
            print(error_message)
            raise Exception(error_message) from e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

            # Clean up the temporary image file if it was created
            temp_path = managed_temp_path[0]
            if not timed_out[0] and temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    if debug:
                        print(f"Debug: Removed temporary file: {temp_path}")
                except Exception as cleanup_err:
                    print(f"Warning: Failed to remove temporary file {temp_path}: {cleanup_err}")


class ExpoLmstudioImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "Describe this image in detail"}),
                "system_prompt": ("STRING", {"default": "This is a chat between a user and an assistant. The assistant is an expert in describing images, with detail and accuracy"}),
                "model_key": ("STRING", {"default": DEFAULT_VISION}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 1}),
                # Legacy parameters for backward compatibility
                "model": ("STRING", {"default": ""}),  # Old parameter name
                "ip_address": ("STRING", {"default": ""}),  # Legacy HTTP mode
                "port": ("INT", {"default": 0, "min": 0, "max": 65535}),  # Legacy HTTP mode
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)
    FUNCTION = "process_image"
    CATEGORY = "ComfyExpo/I2T"

    @classmethod
    def IS_CHANGED(cls, image, user_prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, model="", ip_address="", port=0):
        m = hashlib.sha256()
        
        m.update(str(user_prompt).encode())
        m.update(str(system_prompt).encode())
        m.update(str(model_key).encode())
        m.update(str(auto_unload).encode())
        m.update(str(unload_delay).encode())
        m.update(str(seed).encode())
        m.update(str(max_tokens).encode())
        m.update(str(temperature).encode())
        m.update(str(debug).encode())
        m.update(str(timeout_seconds).encode())
        m.update(str(model).encode())
        m.update(str(ip_address).encode())
        m.update(str(port).encode())
        
        # Include image hash
        if image is not None:
            image_bytes = np.array(image).tobytes()
            m.update(image_bytes)
        
        return m.hexdigest()

    def process_image(self, image, user_prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, model="", ip_address="", port=0):
        # Normalize debug: accept both bool (BOOLEAN widget) and string (legacy/fallback)
        debug = debug if isinstance(debug, bool) else str(debug).lower() == "true"
        # Handle backward compatibility
        # If legacy parameters are provided, show a deprecation warning and try to use them
        if model and not model_key:
            model_key = model
            if debug:
                print("Debug: Using legacy 'model' parameter as 'model_key'")
        
        if ip_address and port > 0:
            # Legacy HTTP mode detected
            return self._process_image_legacy_http(image, user_prompt, system_prompt, model_key or model, ip_address, port, seed, max_tokens, temperature, debug)
        
        # Fail fast if LM Studio is not reachable
        check_lmstudio_connection()

        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting process_image method")
            print(f"Debug: User prompt: {user_prompt}")
            print(f"Debug: Requested Model: {model_key}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")
            print(f"Debug: Image shape: {image.shape}")

        managed_temp_path = [None]
        timed_out = [False]

        def do_the_work():
            with lms.Client() as client:
                # Get model info with fallback
                model_key_to_use = get_model_info_with_fallback(model_key, debug)

                # Get model with proper context management
                if model_key_to_use:
                    if auto_unload == "True" and unload_delay > 0:
                        model_obj = client.llm.model(model_key_to_use, ttl=unload_delay)
                    else:
                        model_obj = client.llm.model(model_key_to_use)
                else:
                    # Use default model
                    model_obj = client.llm.model()

                # Create chat and attach prompts
                chat = lms.Chat(system_prompt)

                # Prepare image and add to chat
                if image is not None:
                    pil_image = Image.fromarray(np.uint8(image[0] * 255))
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        managed_temp_path[0] = tmp.name
                        pil_image.save(managed_temp_path[0], format='JPEG')
                    if debug:
                        print(f"Debug: Saved image to temporary file: {managed_temp_path[0]}")

                    # Use client.files.prepare_image
                    image_handle = client.files.prepare_image(managed_temp_path[0])
                    chat.add_user_message(user_prompt, images=[image_handle])
                else:
                    chat.add_user_message(user_prompt)

                # Configure generation parameters
                config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "seed": seed
                }

                if debug:
                    print(f"Debug: Sending request to LM Studio with config: {config}")

                result = model_obj.respond(chat, config=config)

                output_text = strip_thinking_tags(result.content)

                if debug:
                    try:
                        print(f"Debug: Response received: {output_text[:100]}...")
                    except Exception:
                        print("Debug: Response received (unable to slice content)")

                stats_info = safe_get_stats_info(result, debug)
                if debug:
                    print(f"Debug: Tokens generated: {stats_info['predicted_tokens']}, Time to first token: {stats_info['time_to_first_token']}s")

                # Unload model if requested
                if auto_unload == "True" and unload_delay == 0:
                    try:
                        if debug:
                            print("Debug: Unloading model immediately.")
                        model_obj.unload()
                    except Exception as unload_err:
                        print(f"Warning: Failed to unload model: {unload_err}")

                return (output_text,)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(do_the_work)
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            timed_out[0] = True
            error_message = f"Error: LM Studio operation timed out after {timeout_seconds} seconds. The connection may be unstable."
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"LM Studio error (Image to Text node): {str(e)}"
            print(error_message)
            raise Exception(error_message) from e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

            # Clean up the temporary image file if it was created
            temp_path = managed_temp_path[0]
            if not timed_out[0] and temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    if debug:
                        print(f"Debug: Removed temporary file: {temp_path}")
                except Exception as cleanup_err:
                    print(f"Warning: Failed to remove temporary file {temp_path}: {cleanup_err}")

    def _process_image_legacy_http(self, image, user_prompt, system_prompt, model, ip_address, port, seed, max_tokens=1000, temperature=0.7, debug=False):
        """Legacy HTTP-based image processing for backward compatibility"""
        print("Warning: Using legacy HTTP mode. Consider upgrading to SDK mode for better performance.")
        
        try:
            import requests
        except ImportError:
            return ("Error: requests library not found. Please install it using: pip install requests",)
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting legacy HTTP process_image method")
            print(f"Debug: Text input: {user_prompt}")
            print(f"Debug: Model: {model}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Image shape: {image.shape}")

        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(np.uint8(image[0]*255))

            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Prepare the payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ]}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "seed": seed
            }

            if debug:
                print(f"Debug: Payload prepared, attempting to connect to server")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            }

            url = f"http://{ip_address}:{port}/v1/chat/completions"
            response = requests.post(url, json=payload, headers=headers, timeout=60)

            if debug:
                print(f"Debug: Server response status code: {response.status_code}")

            response.raise_for_status()

            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                generated_text = response_json['choices'][0]['message']['content']
            else:
                generated_text = "No content in the response"

            if debug:
                print(f"Debug: Generated text: {generated_text[:100]}...")

            return (generated_text,)

        except Exception as e:
            error_message = f"Legacy HTTP Error: {str(e)}"
            print(error_message)
            return (error_message,)


class ExpoLmstudioTextGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate a creative story:"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "model_key": ("STRING", {"default": DEFAULT_LLM}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 1}),
                # Legacy parameters for backward compatibility
                "model": ("STRING", {"default": ""}),  # Old parameter name
                "ip_address": ("STRING", {"default": ""}),  # Legacy HTTP mode
                "port": ("INT", {"default": 0, "min": 0, "max": 65535}),  # Legacy HTTP mode
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "generate_text"
    CATEGORY = "ComfyExpo/Text"

    @classmethod
    def IS_CHANGED(cls, prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, model="", ip_address="", port=0):
        m = hashlib.sha256()
        
        m.update(str(prompt).encode())
        m.update(str(system_prompt).encode())
        m.update(str(model_key).encode())
        m.update(str(auto_unload).encode())
        m.update(str(unload_delay).encode())
        m.update(str(seed).encode())
        m.update(str(max_tokens).encode())
        m.update(str(temperature).encode())
        m.update(str(debug).encode())
        m.update(str(timeout_seconds).encode())
        m.update(str(model).encode())
        m.update(str(ip_address).encode())
        m.update(str(port).encode())
        
        return m.hexdigest()

    def generate_text(self, prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, model="", ip_address="", port=0):
        # Normalize debug: accept both bool (BOOLEAN widget) and string (legacy/fallback)
        debug = debug if isinstance(debug, bool) else str(debug).lower() == "true"
        # Handle backward compatibility
        # If legacy parameters are provided, show a deprecation warning and try to use them
        if model and not model_key:
            model_key = model
            if debug:
                print("Debug: Using legacy 'model' parameter as 'model_key'")
        
        if ip_address and port > 0:
            # Legacy HTTP mode detected
            return self._generate_text_legacy_http(prompt, system_prompt, model_key or model, ip_address, port, seed, max_tokens, temperature, debug)
        
        # Fail fast if LM Studio is not reachable
        check_lmstudio_connection()

        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting generate_text method")
            print(f"Debug: Prompt: {prompt}")
            print(f"Debug: Requested Model: {model_key}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")
            print(f"Debug: Max tokens: {max_tokens}")
            print(f"Debug: Temperature: {temperature}")

        def do_the_work():
            with lms.Client() as client:
                # Get model info with fallback
                model_key_to_use = get_model_info_with_fallback(model_key, debug)

                # Get model with proper context management
                if model_key_to_use:
                    if auto_unload == "True" and unload_delay > 0:
                        model_obj = client.llm.model(model_key_to_use, ttl=unload_delay)
                    else:
                        model_obj = client.llm.model(model_key_to_use)
                else:
                    # Use default model
                    model_obj = client.llm.model()

                # Create a new chat
                chat = lms.Chat(system_prompt)

                # Add user message
                chat.add_user_message(prompt)

                # Configure generation parameters
                config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "seed": seed
                }

                if debug:
                    print(f"Debug: Sending request to LM Studio")

                result = model_obj.respond(chat, config=config)

                output_text = strip_thinking_tags(result.content)

                if debug:
                    print(f"Debug: Response received: {output_text[:100]}...")  # Print first 100 characters

                # Extract and log stats information
                stats_info = safe_get_stats_info(result, debug)
                if debug:
                    print(f"Debug: Tokens generated: {stats_info['predicted_tokens']}, Time to first token: {stats_info['time_to_first_token']}s")

                # Unload model immediately if requested (TTL handles delayed unloading)
                if auto_unload == "True" and unload_delay == 0:
                    try:
                        if debug:
                            print(f"Debug: Unloading model immediately.")
                        model_obj.unload()
                    except Exception as unload_err:
                        print(f"Warning: Failed to unload model: {unload_err}")

                return (output_text,)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(do_the_work)
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            error_message = f"Error: LM Studio operation timed out after {timeout_seconds} seconds. The connection may be unstable."
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"LM Studio error (Text Generation node): {str(e)}"
            print(error_message)
            raise Exception(error_message) from e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _generate_text_legacy_http(self, prompt, system_prompt, model, ip_address, port, seed, max_tokens=1000, temperature=0.7, debug=False):
        """Legacy HTTP-based text generation for backward compatibility"""
        print("Warning: Using legacy HTTP mode. Consider upgrading to SDK mode for better performance.")
        
        try:
            import requests
        except ImportError:
            return ("Error: requests library not found. Please install it using: pip install requests",)
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting legacy HTTP generate_text method")
            print(f"Debug: Prompt: {prompt}")
            print(f"Debug: Model: {model}")
            print(f"Debug: System prompt: {system_prompt}")

        try:
            # Prepare the payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "seed": seed
            }

            if debug:
                print(f"Debug: Payload prepared, attempting to connect to server")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            }

            url = f"http://{ip_address}:{port}/v1/chat/completions"
            response = requests.post(url, json=payload, headers=headers, timeout=60)

            if debug:
                print(f"Debug: Server response status code: {response.status_code}")

            response.raise_for_status()

            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                generated_text = response_json['choices'][0]['message']['content']
            else:
                generated_text = "No content in the response"

            if debug:
                print(f"Debug: Generated text: {generated_text[:100]}...")

            return (generated_text,)

        except Exception as e:
            error_message = f"Legacy HTTP Error: {str(e)}"
            print(error_message)
            return (error_message,)


class ExpoLmstudioStructuredOutput:
    """
    LM Studio Structured Output node for ComfyUI.

    Sends a prompt (and optional image) to an LM Studio model and enforces a
    JSON response that matches the provided JSON Schema.  The full JSON string
    is emitted on the first output pin; up to six individual key values are
    extracted and emitted on value_1 … value_6 according to the newline-
    separated list in ``output_keys``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image.",
                    "tooltip": "The user message / prompt sent to the model.",
                }),
                "json_schema": ("STRING", {
                    "multiline": True,
                    "default": '{\n  "type": "object",\n  "properties": {\n    "subject": {"type": "string"},\n    "style": {"type": "string"},\n    "mood": {"type": "string"},\n    "tags": {"type": "array", "items": {"type": "string"}}\n  },\n  "required": ["subject", "style", "mood", "tags"]\n}',
                    "tooltip": "A valid JSON Schema object. The model will be constrained to return JSON matching this schema.",
                }),
                "output_keys": ("STRING", {
                    "multiline": True,
                    "default": "subject\nstyle\nmood\ntags",
                    "tooltip": "Newline-separated list of top-level JSON keys to extract into value_1 … value_6 outputs (in order). Array values are joined with ', '.",
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful AI assistant. Always respond with valid JSON.",
                }),
                "model_key": ("STRING", {"default": DEFAULT_LLM}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("json_output", "value_1", "value_2", "value_3", "value_4", "value_5", "value_6")
    FUNCTION = "generate_structured"
    CATEGORY = "ComfyExpo/LMStudio"

    @classmethod
    def IS_CHANGED(cls, text_input, json_schema, output_keys, system_prompt, model_key,
                   auto_unload, unload_delay, seed, image=None,
                   max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300):
        import json as _json
        m = hashlib.sha256()
        for val in (text_input, json_schema, output_keys, system_prompt, model_key,
                    auto_unload, unload_delay, seed, max_tokens, temperature, debug, timeout_seconds):
            m.update(str(val).encode())
        if image is not None:
            m.update(np.array(image).tobytes())
        return m.hexdigest()

    def generate_structured(self, text_input, json_schema, output_keys, system_prompt,
                            model_key, auto_unload, unload_delay, seed,
                            image=None, max_tokens=1000, temperature=0.7,
                            debug=False, timeout_seconds=300):
        import json as _json

        debug = debug if isinstance(debug, bool) else str(debug).lower() == "true"
        check_lmstudio_connection()

        # Parse the schema
        try:
            parsed_schema = _json.loads(json_schema)
        except _json.JSONDecodeError as exc:
            raise Exception(f"Structured Output: invalid JSON Schema — {exc}") from exc

        # Parse requested output keys (up to 6)
        keys = [k.strip() for k in output_keys.splitlines() if k.strip()][:6]

        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: [StructuredOutput] model={model_key}, keys={keys}")
            print(f"Debug: [StructuredOutput] schema={_json.dumps(parsed_schema, indent=2)}")

        managed_temp_path = [None]
        timed_out = [False]

        def do_the_work():
            with lms.Client() as client:
                model_key_to_use = get_model_info_with_fallback(model_key, debug)

                if model_key_to_use:
                    if auto_unload == "True" and unload_delay > 0:
                        model_obj = client.llm.model(model_key_to_use, ttl=unload_delay)
                    else:
                        model_obj = client.llm.model(model_key_to_use)
                else:
                    model_obj = client.llm.model()

                chat = lms.Chat(system_prompt)

                if image is not None:
                    pil_image = Image.fromarray(np.uint8(image[0] * 255))
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        managed_temp_path[0] = tmp.name
                        pil_image.save(managed_temp_path[0], format='JPEG')
                    image_handle = client.files.prepare_image(managed_temp_path[0])
                    chat.add_user_message(text_input, images=[image_handle])
                    if debug:
                        print(f"Debug: [StructuredOutput] added image + text to chat")
                else:
                    chat.add_user_message(text_input)

                # Build config with structured output constraint
                config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "seed": seed,
                    "structured": {
                        "type": "json",
                        "jsonSchema": parsed_schema,
                    },
                }

                if debug:
                    print(f"Debug: [StructuredOutput] sending request with structured config")

                result = model_obj.respond(chat, config=config)

                json_string = strip_thinking_tags(result.content)

                if debug:
                    print(f"Debug: [StructuredOutput] raw response: {json_string[:200]}")

                # Parse the returned JSON and extract key values
                try:
                    parsed = _json.loads(json_string)
                except _json.JSONDecodeError:
                    # Model may have wrapped JSON in a code fence -- strip it
                    cleaned = json_string
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("```", 1)[0]
                    try:
                        parsed = _json.loads(cleaned.strip())
                        json_string = cleaned.strip()
                        if debug:
                            print("Debug: [StructuredOutput] JSON parsed after stripping code fence")
                    except _json.JSONDecodeError as exc2:
                        if debug:
                            print(f"Debug: [StructuredOutput] JSON parse failed: {exc2}")
                        parsed = {}

                def _to_str(val):
                    """Convert a JSON value to a plain string."""
                    if isinstance(val, list):
                        return ", ".join(str(v) for v in val)
                    if isinstance(val, dict):
                        return _json.dumps(val)
                    return str(val) if val is not None else ""

                values = []
                for k in keys:
                    values.append(_to_str(parsed.get(k, "")))
                # Pad to 6 slots
                while len(values) < 6:
                    values.append("")

                stats_info = safe_get_stats_info(result, debug)
                if debug:
                    print(f"Debug: [StructuredOutput] tokens={stats_info['predicted_tokens']}, ttft={stats_info['time_to_first_token']}s")

                if auto_unload == "True" and unload_delay == 0:
                    try:
                        model_obj.unload()
                    except Exception as unload_err:
                        print(f"Warning: Failed to unload model: {unload_err}")

                return (json_string, values[0], values[1], values[2], values[3], values[4], values[5])

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(do_the_work)
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            timed_out[0] = True
            err = f"Error: LM Studio operation timed out after {timeout_seconds} seconds. The connection may be unstable."
            print(err)
            empty = ("",) * 6
            return (err,) + empty
        except Exception as e:
            error_message = f"LM Studio error (Structured Output node): {str(e)}"
            print(error_message)
            raise Exception(error_message) from e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

            # Clean up the temporary image file if it was created
            temp_path = managed_temp_path[0]
            if not timed_out[0] and temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_err:
                    print(f"Warning: Failed to remove temporary file {temp_path}: {cleanup_err}")


NODE_CLASS_MAPPINGS = {
    "ExpoLmstudioUnified": ExpoLmstudioUnified,
    "ExpoLmstudioImageToText": ExpoLmstudioImageToText,
    "ExpoLmstudioTextGeneration": ExpoLmstudioTextGeneration,
    "ExpoLmstudioStructuredOutput": ExpoLmstudioStructuredOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExpoLmstudioUnified": "Expo LM Studio Unified",
    "ExpoLmstudioImageToText": "Expo LM Studio Image to Text",
    "ExpoLmstudioTextGeneration": "Expo LM Studio Text Generation",
    "ExpoLmstudioStructuredOutput": "Expo LM Studio Structured Output",
}

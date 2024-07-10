# LM Studio Image to Text Node for ComfyUI

This custom node for ComfyUI allows you to use LM Studio's vision models to generate text descriptions of images. It's designed to work with LM Studio's local API, providing a flexible and customizable way to integrate image-to-text capabilities into your ComfyUI workflows.

## Workflow Example

Here's an example of how the LM Studio Image to Text node can be used in a ComfyUI workflow:

![LM Studio Image to Text Workflow](workflow.png)

## Features

- Generate text descriptions of images using LM Studio's vision models
- Customizable system prompt
- Flexible model selection
- Configurable server address and port
- Debug mode for troubleshooting

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
   ```
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/your-username/lmstudio-image-to-text-node.git
   ```
2. Restart ComfyUI or reload custom nodes.

## Usage

Add the "LM Studio Image To Text" node to your ComfyUI workflow. Connect an image output to the "image" input of the node.

### Inputs

- **image** (required): The input image to be described.
- **text_input** (required): The prompt for the image description. Default: "What's in this image?"
- **model** (required): The name of the LM Studio model to use. While this is a required field in ComfyUI, you don't have to change it if you're using the default model. It's useful to set this when sharing workflows to ensure others use the same model. Default: "billborkowski/llava-NousResearch_Nous-Hermes-2-Vision-GGUF"
- **system_prompt** (required): The system prompt to set the context for the AI. Default: "This is a chat between a user and an assistant. The assistant is helping the user to describe an image."
- **ip_address** (required): The IP address of your LM Studio server. Default: "localhost"
- **port** (required): The port number of your LM Studio server. Default: 1234
- **debug** (optional): Set to True for detailed logging. Default: False

### Output

- **Description**: The generated text description of the input image.

## LM Studio Setup

1. Install and run LM Studio on your local machine.
2. Load a vision model in LM Studio (e.g., "billborkowski/llava-NousResearch_Nous-Hermes-2-Vision-GGUF").
3. Ensure the LM Studio API is running and accessible (default: http://localhost:1234).

## Notes

- This node is adapted from the LM Studio code example for image-to-text conversion.
- The `model` parameter doesn't have to be changed from its default value, but it's useful to set it explicitly when sharing workflows to ensure consistency.
- Adjust the `ip_address` and `port` if your LM Studio instance is running on a different machine or port.

## Original LM Studio Code

This node is adapted from the following LM Studio code example:

```python
# Adapted from OpenAI's Vision example
from openai import OpenAI
import base64
import requests

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Ask the user for a path on the filesystem:
path = input("Enter a local filepath to an image: ")

# Read the image and encode it to base64:
base64_image = ""
try:
    image = open(path.replace("'", ""), "rb").read()
    base64_image = base64.b64encode(image).decode("utf-8")
except:
    print("Couldn't read the image. Make sure the path is correct and the file exists.")
    exit()

completion = client.chat.completions.create(
    model="moondream/moondream2-gguf",
    messages=[
        {
            "role": "system",
            "content": "This is a chat between a user and an assistant. The assistant is helping the user to describe an image.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ],
    max_tokens=1000,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Troubleshooting

If you encounter any issues:
1. Enable debug mode by setting the `debug` input to True.
2. Check the ComfyUI console for error messages and debug output.
3. Verify that LM Studio is running and accessible at the specified IP address and port.
4. Ensure you're using a compatible vision model in LM Studio.

For further assistance, please open an issue on the GitHub repository.
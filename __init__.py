from .expo_lmstudio_imagetotext import ExpoLmstudioImageToText, ExpoLmstudioTextGeneration


# Update these in your __init__.py file
NODE_CLASS_MAPPINGS = {
    "LM Studio Image To Text": ExpoLmstudioImageToText,
    "LM Studio Text Generation": ExpoLmstudioTextGeneration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LM Studio Image To Text": "Expo LMStudio Image to Text",
    "LM Studio Text Generation": "Expo LMStudio Text Generation"
}
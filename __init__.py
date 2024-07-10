from .expo_lmstudio_imagetotext import ExpoLmstudioImageToText


# This goes in your __init__.py file
NODE_CLASS_MAPPINGS = {
    "LM Studio Image To Text": ExpoLmstudioImageToText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LM Studio Image To Text": "Expo LMStudio Image to Text"
}
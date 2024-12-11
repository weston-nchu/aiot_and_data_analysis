import requests
from transformers import pipeline

# Function to translate Chinese sentence to English using Hugging Face API
def translate_to_english(chinese_sentence):
    translation_model = pipeline("translation_zh_to_en")
    english_sentence = translation_model(chinese_sentence)[0]['translation_text']
    return english_sentence

# Function to generate image from English sentence using Hugging Face API
def generate_image_from_text(text):
    # Use Hugging Face API for image generation model (e.g., DALL·E)
    api_url = "https://api-inference.huggingface.co/models/dalle-mega-1"
    headers = {"Authorization": "Bearer YOUR_HUGGINGFACE_API_KEY"}
    data = {"inputs": text}
    
    response = requests.post(api_url, headers=headers, json=data)
    
    if response.status_code == 200:
        image_url = response.json()[0]["generated_image"]
        return image_url
    else:
        print(f"Error generating image: {response.status_code}")
        return None

# Main function to execute translation and image generation
def main():
    chinese_sentence = input("Enter a Chinese sentence: ")
    
    # Step 1: Translate Chinese to English
    english_sentence = translate_to_english(chinese_sentence)
    print(f"Translated to English: {english_sentence}")
    
    # Step 2: Generate an image based on the translated English sentence
    image_url = generate_image_from_text(english_sentence)
    
    if image_url:
        print(f"Image generated: {image_url}")

if __name__ == "__main__":
    main()

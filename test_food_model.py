
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "nateraw/food"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

model.to(device)
model.eval()

test_images = [
    "apple.jpg",
    "banana.jpg",
    "basmati-rice.jpg",
    "biriyaniu.jpg",
    "chicken breast.jpg",
    "salad.jpg"
]

for filename in test_images:
    image_path = os.path.join("static", "uploads", filename)

    print("\n" + "=" * 45)
    print("Image:", filename)

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        top_results = torch.topk(probabilities, 3)

        for score, index in zip(
            top_results.values[0], top_results.indices[0]
        ):
            label = model.config.id2label[index.item()]
            print(f"{label}: {score.item() * 100:.2f}%")

    except Exception as e:
        print("Error:", e)

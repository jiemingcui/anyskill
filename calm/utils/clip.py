from PIL import Image
import requests
import time

from transformers import CLIPProcessor, CLIPModel

start = time.time()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

end1 = time.time()
print("Time of model loading is ", (end1 - start))
image = Image.open("/home/cjm/Videos/sample.jpg")
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(text=["pink"], images=image, return_tensors="pt", padding=True)
inputs = processor(text=["laying", "drawing", "playing violin", "runing", "getting up", "dancing"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)

end2 = time.time()
print("Time of model running is ", (end2 - start))

logits_per_image = outputs.logits_per_image # this is the image-text similarity score
print(logits_per_image)
end3 = time.time()
print("Time of similarity calculation is ", (end3 - end1))

probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
end4 = time.time()
print("Time of softmax is ", (end4 - start))




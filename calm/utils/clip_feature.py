from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

class CLIPFeature(nn.Module):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        super().__init__()
        self.image = self.rendered_image
        # image = Image.open("./images_256/180.png")
        # image = Image.open(requests.get(url, stream=True).raw)
        self.text = self.language_input

    def similarity(self):
        inputs = processor(text=self.text, images=self.image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        print(0)

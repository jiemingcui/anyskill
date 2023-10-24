# import torch
# from PIL import Image
# import open_clip
#
# # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip/b32_400m.pt')
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='./b32_400m.pt')
#
# image = preprocess(Image.open("/home/cjm/Videos/sample.jpg")).unsqueeze(0)
# text = open_clip.tokenize(["laying", "drawing", "playing violin", "runing", "getting up"])
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#
#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#
# print("Label probs:", text_probs)
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("/home/cjm/Videos/sample.jpg")).unsqueeze(0)
text = tokenizer(["laying", "drawing", "playing violin", "runing", "getting up", "dancing"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = image_features @ text_features.T - 0.22
    # text_probs = torch.exp(-2*text_probs)
    # text_probs = 100.0 * image_features @ text_features.T
    # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
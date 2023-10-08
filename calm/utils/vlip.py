from PIL import Image
import requests
# from utils.device_dtype_mixin import DeviceDtypeModuleMixin
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import cv2
import numpy as np
import torch

# Code to convert one video to few images.
def video2image(video_path, frame_rate=1.0, size=224):
    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)

    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        images = np.zeros([3, size, size], dtype=np.float32)
        print("ERROR: problem reading video file: ", video_path)
    else:
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        interval = fps / frame_rate
        frames_idx = np.floor(np.arange(start_sec * fps, end_sec * fps, interval))
        ret = True
        images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)

        for i, idx in enumerate(frames_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = i
            images[i, :, :, :] = preprocess(size, Image.fromarray(frame).convert("RGB"))

        images = images[:last_frame + 1]
    cap.release()
    video_frames = torch.tensor(images)
    return video_frames

caps = ["dancing", "ballet", "a green agent", "running", "step forward"]
video_seq = video2image("/home/cjm/Videos/ballet.mp4")

vlip_text_model = CLIPTextModelWithProjection.from_pretrained("./clip4clip_huggingface")
vlip_vis_model = CLIPVisionModelWithProjection.from_pretrained("./clip4clip_huggingface")
vlip_tokenizer = CLIPTokenizer.from_pretrained("./clip4clip_huggingface")


vis_emb = vlip_vis_model(video_seq)
vis_emb = vis_emb["image_embeds"]
vis_emb = vis_emb / vis_emb.norm(dim=-1, keepdim=True)
vis_emb = torch.mean(vis_emb, dim=0)
vis_emb = vis_emb / vis_emb.norm(dim=-1, keepdim=True)

for cap in caps:
    inputs = vlip_tokenizer(text=cap, return_tensors="pt")
    text_emb = vlip_text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    text_emb = text_emb[0] / text_emb[0].norm(dim=-1, keepdim=True)

    vis_emb_np = vis_emb.detach().numpy()
    text_emb_np = text_emb.detach().numpy()

    sim_matrix = torch.matmul(vis_emb, text_emb.t())
    sim_matrix_np = np.matmul(vis_emb_np, text_emb_np.transpose())
    print(cap, "   ", sim_matrix_np)
    # print(0)
# from transformers import CLIPVisionModelWithProjection
#
#
# model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
# model = model.eval()
# visual_output = model(video)
#
# # Normalizing the embeddings and calculating mean between all embeddings.
# visual_output = visual_output["image_embeds"]
# visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
# visual_output = torch.mean(visual_output, dim=0)
# visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
# print(visual_output)




# class VLIP(DeviceDtypeModuleMixin):
#
#
#
#
#     def __init__(self):
#
#
#
#
#
#         super.__init__()
#
#     # def __init__(self):
#     #     self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     #     self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     #     super().__init__()
#     #     self.image = self.rendered_image
#     #     # image = Image.open("./images_256/180.png")
#     #     # image = Image.open(requests.get(url, stream=True).raw)
#     #     self.text = self.language_input
#     #
#     # def similarity(self):
#     #     inputs = processor(text=self.text, images=self.image, return_tensors="pt", padding=True)
#     #     outputs = self.model(**inputs)
#     #     logits_per_image = outputs.logits_per_image # this is the image-text similarity score
#     #     # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
#     #     print(0)




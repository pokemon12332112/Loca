from models.loca import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser
from PIL import Image
import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torchvision import transforms as T
import json

with open(r'D:\CSAM\FSC147\annotation_FSC147_384.json') as f:
    annotations = json.load(f)
with open(r'D:\CSAM\data.json') as f:
    prediction = json.load(f)
with open(r'D:\CSAM\FSC147\Train_Test_Val_FSC_147.json') as f:
    data_split = json.load(f)
im_ids_train = data_split['train']
im_ids_test = data_split['test']
im_ids_val = data_split['val']

gpu = 'cuda'
torch.cuda.set_device(0)
device = torch.device(gpu)

model = build_model().to(device),
model = model[0]

state_dict = torch.load(r'D:\CSAM\loca-main\loca_few_shot.pt')['model']

model.load_state_dict(state_dict)
model.eval()
img_size = 512
img_path = r'D:\CSAM\FSC147\images_384_VarV2\1123.jpg'
img = Image.open(img_path).convert("RGB")
new_h = 384
new_w = 16 * (int(384 / 16))
img = img.resize((new_h, new_w))
w, h = img.size

img = T.Compose([
    T.ToTensor(),
    T.Resize((img_size, img_size)),
])(img)
all_bboxes = []
for box in prediction['1123.jpg']:
    bboxes = torch.tensor(
        [box[1], box[0], box[3], box[2]],
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)
    bboxes = bboxes / torch.tensor([w, h, w, h]) * img_size
    all_bboxes.append(bboxes)
# bboxes = torch.tensor(
#     prediction['1123.jpg'],
#     dtype=torch.float32
# )
img = img.to(device)
preds = []
for bboxes in all_bboxes:
    bboxes = all_bboxes[0].to(device)
    out, _ = model(img, bboxes)
    pred = out.flatten(1).sum()
    preds.append(pred)
# bboxes = all_bboxes[0].to(device)
# out, _ = model(img, bboxes)
# print(out.flatten(1).sum())
print(pred)
# print(all_bboxes[0])

# bboxes = bboxes / torch.tensor([w, h, w, h]) * img_size

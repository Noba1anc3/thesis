import torch
import os
import sys
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision
sys.path.insert(0, '/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/')

a = torch.load("/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_dev_layoutlm-base-uncased_512_sen")
b = torch.load("/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_dev_layoutlm-base-uncased_512_tok")

for i in range(len(a)):
    b[i].row_lines = a[i].boxes
torch.save(b, "/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_dev_layoutlm-base-uncased_512_tok")
print(1)
# image = a[0].resized_image
# image = ToTensor()(image).unsqueeze(0)
# model = torchvision.models.resnet101(pretrained=True)
# backbone = nn.Sequential(*(list(model.children())[:-3]))
# feature_maps = backbone(image)

print(1)
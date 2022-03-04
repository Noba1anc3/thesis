import torch
import os
import sys
sys.path.insert(0, '/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/')

a = torch.load("/home/dreamaker/thesis/thesis/layoutlms/layoutlm/deprecated/examples/seq_labeling/data/cached_train_layoutlm-base-uncased_512")
image = a[0].resized_image
bboxes = a[0].resized_and_aligned_bboxes
image.show('1')
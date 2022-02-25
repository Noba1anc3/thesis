
import os
import torch
__dir__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
import sys
print(__dir__)
sys.path.append(__dir__)


# a = torch.load("unilm/layoutlm/deprecated/examples/seq_labeling/data/cached_train_layoutlm-base-uncased_512")
# print(1)

import shutil

list_dirs = os.listdir('.')
name = ''
print(list_dirs)
for item in list_dirs:
    if item.find("checkpoint") != -1:
        name = item
        break
print(name)
try:
    shutil.rmtree(name)
except:
    pass
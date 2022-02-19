import json
import os

dataset = ['KOMAGUSAM', 'NETAPP', 'RUCKSWIRE', 'ACTMAX', 'ADAMP', 'AKRIBIS', 'ALPHA', 'AMPHENOLE',
           'Cisco', 'INFINEON', 'Micron', 'SIIX', 'TI', 'TOSHIBA']
nd = ['air_waybill']
keylist = []

import torch
a = torch.load('cached_train_layoutlm-base-uncased_512')

for set in dataset[7:8]:
    keylist.append([set])
    dataset_folder = os.path.join('Singapore Customs Dataset', set, 'test/json')
    i = 0
    for file in sorted(os.listdir(dataset_folder)):
        file_path = os.path.join(dataset_folder, file)
        file_json = json.load(open(file_path))
        file_json1 = file_json['single'][0]['res']['items']
        for item in file_json1:
            if str(item.keys())[12:-3] == 'INVNW':
                print(file_path, i)
                # item["ignore"] = item.pop("INVNo2")
                # print(file_path, i)
                # i += 1
                # new_json = json.dumps(file_json)
                # with open(file_path, 'w', encoding='utf-8') as f:
                #     f.write(new_json)
                #     f.close()
            if not str(item.keys())[12:-3] in keylist[-1]:
                keylist[-1].append(str(item.keys())[12:-3])

allkey = []
for key in keylist:
    print(len(key),key)
    for subkey in key:
        if not subkey in allkey and not subkey in dataset:
            allkey.append(subkey)

print(allkey)




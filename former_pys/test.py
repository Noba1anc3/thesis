# path = 'output/TI/train_image.txt'
#
# with open(path, 'r') as f:
#     a = f.readlines()
#
#
#
# for i, line in enumerate(a):
#     if line != '\n':
#         name = line.split('\t')[3]
#         print(name)

path = 'Singapore Customs Dataset'

dataset = ['KOMAGUSAM', 'NETAPP', 'RUCKSWIRE', 'ACTMAX', 'ADAMP', 'AKRIBIS', 'ALPHA', 'AMPHENOLE',
           'CISCO', 'INFINEON', 'MICRON', 'SIIX', 'TI', 'TOSHIBA']

numset = [331, 120, 222, 112, 111, 90, 315, 112, 112, 99, 57, 112, 112, 36]

import os
from shutil import copyfile

for i, set in enumerate(dataset):
    newpath = os.path.join(path, set, 'invoice/json')
    trainPath = os.path.join(path, set, 'train')
    testPath = os.path.join(path, set, 'test')
    overalltrainPath = os.path.join(path, 'SG Dataset', 'train')
    overalltestPath = os.path.join(path, 'SG Dataset', 'test')

    if not os.path.exists(trainPath):
        os.mkdir(trainPath)
    if not os.path.exists(testPath):
        os.mkdir(testPath)
    if not os.path.exists(overalltestPath):
        os.mkdir(overalltestPath)
    if not os.path.exists(overalltrainPath):
        os.mkdir(overalltrainPath)

    if not os.path.exists(os.path.join(trainPath, 'image')):
        os.mkdir(os.path.join(trainPath, 'image'))
    if not os.path.exists(os.path.join(trainPath, 'json')):
        os.mkdir(os.path.join(trainPath, 'json'))

    if not os.path.exists(os.path.join(testPath, 'image')):
        os.mkdir(os.path.join(testPath, 'image'))
    if not os.path.exists(os.path.join(testPath, 'json')):
        os.mkdir(os.path.join(testPath, 'json'))

    if not os.path.exists(os.path.join(overalltrainPath, 'image')):
        os.mkdir(os.path.join(overalltrainPath, 'image'))
    if not os.path.exists(os.path.join(overalltrainPath, 'json')):
        os.mkdir(os.path.join(overalltrainPath, 'json'))

    if not os.path.exists(os.path.join(overalltestPath, 'image')):
        os.mkdir(os.path.join(overalltestPath, 'image'))
    if not os.path.exists(os.path.join(overalltestPath, 'json')):
        os.mkdir(os.path.join(overalltestPath, 'json'))

    image_path = newpath[:-4] + 'image'

    num = numset[i]

    for index in range(len(sorted(os.listdir(newpath)))):
        jsonFile = sorted(os.listdir(newpath))[index]
        imgFile = sorted(os.listdir(image_path))[index]

        detailedJson = os.path.join(newpath, jsonFile)
        detailedImg = os.path.join(image_path, imgFile)

        newjson = os.path.join(trainPath, 'json', jsonFile)
        newImg = os.path.join(trainPath, 'image', imgFile)

        allJson = os.path.join(overalltrainPath, 'json', jsonFile)
        allImg = os.path.join(overalltrainPath, 'image', imgFile)

        if index < num:
            # copyfile(detailedJson, newjson)
            # copyfile(detailedImg, newImg)
            # copyfile(detailedJson, allJson)
            copyfile(detailedImg, allImg)
        else:
            newjson = os.path.join(testPath, 'json', jsonFile)
            newImg = os.path.join(testPath, 'image', imgFile)

            allJson = os.path.join(overalltestPath, 'json', jsonFile)
            allImg = os.path.join(overalltestPath, 'image', imgFile)

            # copyfile(detailedJson, newjson)
            # copyfile(detailedImg, newImg)
            # copyfile(detailedJson, allJson)
            copyfile(detailedImg, allImg)


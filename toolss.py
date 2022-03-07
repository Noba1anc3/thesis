

import os

from methods import *
from Direction_Classify.tools.infer.correct import rotate_bound

all_datas = "SG_Dataset"


train_img_path = os.path.join( all_datas, "train", "image")
train_jsn_path = os.path.join( all_datas, "train", "json")

test_img_path = os.path.join( all_datas, "test", "image")
test_jsn_path = os.path.join( all_datas, "test", "json")


colors = sem_colors()


test = 1

train_test_jsn = [train_jsn_path, test_jsn_path]
train_test_img = [train_img_path, test_img_path]

for i, file in enumerate(os.listdir(train_test_jsn[test])):
    # img_path = os.path.join(train_test_img[test], file[:-4] + 'png')
    # img = get_img_ori(img_path)
    jsn = open_json(os.path.join(train_test_jsn[test], file))
    new_jsn = tokenize(jsn)

    
    # img, ratio = get_img(img_path)

    # for j in range(len(jsn["items"])-1, -1, -1):
    #     item = jsn["items"][j]
    #     key = list(item.keys())[0]

    #     loc = item[key]["locations"]
    #     val = item[key]["value"]
    #     # if loc[0][1] >= loc[1][1] or loc[0][0] >= loc[1][0]:
    #     #     print(i, j)
    #         # del jsn["items"][j]
    #     lu, rd = cal_lu_rd_ori(loc, ratio)

    #     cv.rectangle(img, lu, rd, (11,214,14))
        # cv.putText(img, val, lu, cv.FONT_HERSHEY_COMPLEX, 0.4, color=colors[sem_labels.index(key)])

        
        # if val.rstrip() 
        # if threshold0 < abs(loc[0][0] - loc[3][0]) < threshold or threshold0 < abs(loc[0][1] - loc[1][1]) < threshold or \
        # threshold0 < abs(loc[1][0] - loc[2][0]) < threshold or threshold0 < abs(loc[2][1] - loc[3][1]) < threshold:
            # bigs.append(i)
        # lu, rd = cal_lu_rd_origin(loc)

        # if (i, j) not in files: files.append((i, j))
        # lu, rd = cal_lu_rd_ori(loc, wr, hr)
            # print(lu, rd)
        # cv.rectangle(img, lu, rd, colors[sem_labels.index(key)])
            #yes = True
            # if file not in files: files.append(file)

    with open(os.path.join(train_test_jsn[test], file), 'w') as f:
        json.dump(new_jsn, f)

    # cv.imshow(str('1'), img)
    # cv.waitKey(0)

import numpy as np
import os
import cv2
import time


def angle_correction(img):
    curtime = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)  # 二值化
    whereid = np.where(thresh == 0)
    whereid = whereid[::-1]
    coords = np.column_stack(whereid)
    angle = cv2.minAreaRect(coords)[2]
    if angle > 45:
        angle -= 90
    rotate_img, _ = rotate_bound(img, angle)
    return rotate_img, time.time() - curtime


def avg_count(img):
    sum_r = 0
    sum_g = 0
    sum_b = 0
    for i in range(10):
        for j in range(10):
            sum_r += img[i][j][0]
            sum_g += img[i][j][1]
            sum_b += img[i][j][2]
    return sum_r/100, sum_g/100, sum_b/100


def avg(img1, img2, img3, img4):
    r1, g1, b1 = avg_count(img1)
    r2, g2, b2 = avg_count(img2)
    r3, g3, b3 = avg_count(img3)
    r4, g4, b4 = avg_count(img4)
    return (r1+r2+r3+r4)//4, (g1+g2+g3+g4)//4, (b1+b2+b3+b4)//4




def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    ori_M = M
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # print(M)
    avg_r, avg_g, avg_b = avg(image[0:10, 0:10], image[0:10, w-10:w], image[h-10:h, 0:10], image[h-10:h, w-10:w])

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(avg_r, avg_g, avg_b)), ori_M


if __name__ == "__main__":
    src = r'G:\054\datasets\add_test\test'
    des = r'G:\054\datasets\add_test\temp'
    dirlist = os.listdir(src)
    # dirlist = ['dianzishu']
    i = 1
    sum_time = 0
    for dir in dirlist:
        src_dir = os.path.join(src, dir + '\\' + 'images')
        des_dir = os.path.join(des, dir + '\\' + 'images')
        src_txt_path = os.path.join(src, dir + '\\' + 'label.txt')
        des_txt_path = os.path.join(des, dir)
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)

        # shutil.copy(src_txt_path, des_txt_path)
        imglist = os.listdir(src_dir)
        for img_name in imglist:
            start_time = time.time()
            img = cv2.imread(os.path.join(src_dir, img_name))
            rotateImg = angle_correction(img)
            end_time = time.time()
            sum_time += (end_time - start_time)
            # src_img = cv2.imread(os.path.join(src_dir, img))
            # rotateImg = rotate_bound(src_img, angle)
            cv2.imwrite(os.path.join(des_dir, img_name), rotateImg)
            print(i)
            i += 1

    print(sum_time/3600)

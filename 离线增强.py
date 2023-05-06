# -*- coding: utf-8 -*-
"""
@Time: 2023/4/2 21:11
@Auth: 除以七  ➗7️⃣
@File: 离线增强.py
@E-mail: divided.by.07@gmail.com
@Github: https://github.com/divided-by-7
@info: None
"""
import math
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

np.set_printoptions(suppress=True)


# 辅助函数

def cv2show(image, win_name="image", delay=100, is_show=True, close=True, display_windows_size=[1920, 1080],
            scale_rate=0.7, save_image=False):
    if is_show:
        win_name = win_name + ", auto close this window after {} s".format(delay)
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 40, 30)
        windows_size = [0, 0]
        windows_size[0] = int(display_windows_size[0] * 0.7)
        windows_size[1] = int(display_windows_size[1] * 0.7)
        # print("画图的image.shape:",image.shape) # 没变过
        # print("画图的窗口尺寸",windows_size)
        image_copy = image.copy()
        if image.shape[0] > windows_size[1]:
            rate = windows_size[1] / image_copy.shape[0]
            # print("检测到图片高度超出屏幕，自动缩小图片，缩放比例为", rate)
            image_copy = cv2.resize(image, (int(image.shape[1] * rate), int(image.shape[0] * rate)))
        if image_copy.shape[1] > windows_size[0]:
            rate1 = windows_size[0] / image_copy.shape[1]
            # print("检测到图片宽度超出屏幕，自动缩小图片，缩放比例为", rate1)
            image_copy = cv2.resize(image, (int(image.shape[1] * rate1), int(image.shape[0] * rate1)))
        cv2.imshow(win_name, image_copy)
        cv2.waitKey(delay * 1000)
        if save_image:
            cv2.imwrite("win_name" + ".jpg", image)

    if close:
        cv2.destroyAllWindows()


def xywh_rate2xyxy(image, x):
    h, w, _ = image.shape
    # x = label
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)

    y[:, 1] = (x[:, 1] - x[:, 3] / 2) * w  # top left x
    y[:, 2] = (x[:, 2] - x[:, 4] / 2) * h  # top left y
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) * w  # bottom right x
    y[:, 4] = (x[:, 2] + x[:, 4] / 2) * h  # bottom right y

    return y


def xyxy_2xyxy(x, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] += padw  # top left x
    y[:, 1] += padh  # top left y
    y[:, 2] += padw  # bottom right x
    y[:, 3] += padh  # bottom right y
    return y


def draw_box(image, label):
    # 要求label是xyxy格式，且为像素长度而不是比例长度
    image_copy = image.copy()
    for box in label:
        left_top_point = (int(box[1]), int(box[2]))
        right_down_point = (int(box[3]), int(box[4]))
        cv2.rectangle(image_copy, left_top_point, right_down_point, (255, 0, 0), thickness=5)
    return image_copy


def bbox_ioa(box1, box2, eps=1e-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


# ---------------------------------------------------------------------------------------------------
"""不改变Label的增强："""


# def multi_scale(image):
#     # 注：如果是yolo格式的label，xywh都是以归一化形式存储，所以图片整体改变形状不需要改变label
#     # 设置缩放尺度最小为原图的min_scale倍，最大为原图的max_scale倍
#     min_scale = 0.5
#     max_scale = 1.5
#     imgsz = np.array(image.shape[:2])
#     print(imgsz)
#     scale = np.random.uniform(min_scale, max_scale)
#     new_imgsz = np.int64(np.ceil(imgsz * scale))
#     print("scale:", scale)
#     print("new_imgsz", new_imgsz)
#     image = cv2.resize(image, new_imgsz[::-1], interpolation=cv2.INTER_LINEAR)
#     print("reshape尺寸", image.shape)
#     return image


# HSV空间增强
def augment_hsv(image, label, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    # HSV color-space augmentation
    # HSV空间增强
    # HSV空间增强，对HSV做一个随机变化
    # 即对明度、饱和度、色调随机变换
    """
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        # r = [a,b,c], a b c在[0,2]上，gain越小越接近1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        # np.clip : 截断，保证值在0,255区间，如果溢出则为界限值
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img_result = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  # no return needed
        return img_result, label


# 直方图均衡化，使用后能提高图像的对比度
def hist_equalize(image, label, clahe=True, bgr=True):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    # 在BGR图像'im'上用im.shape(n,m,3)和范围0-255均衡化直方图
    # 像素的亮度（Y）以及红色分量与亮度的信号差值（U）和蓝色与亮度的差值（V）
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB), label  # convert YUV image to RGB


# ---------------------------------------------------------------------------------------------------
"""改变label的增强："""


def scale(image, label, imgsz=640):
    max_scale = max(image.shape[0], image.shape[1])
    scale_rate = imgsz / max_scale
    new_imgsz = [int(image.shape[1] * scale_rate), int(image.shape[0] * scale_rate)]
    image = cv2.resize(image, new_imgsz, interpolation=cv2.INTER_LINEAR)
    label[:, 1:] = np.int64(label[:, 1:] * scale_rate)
    return image, label


def replicate(image, label):
    # Replicate labels
    # 复制labels
    h, w = image.shape[:2]
    boxes = label[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        # argsort:从小到大排序，返回索引值
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        label = np.append(label, [[label[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return image, label


# 随机透视、旋转、缩放
def random_perspective(image, label, degrees=10, translate=.1,
                       scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(label)
    if n:
        new = np.zeros((n, 4))

        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=label[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        label = label[i]
        label[:, 1:5] = new[i]

    return image, label


# 随机为图像上添加空白格噪声
def cutout(image, label, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = image.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        # [0.5, 0.25, 0.25, 0.125, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
        # 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125]
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(label) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xyxy_2xyxy(label[:, 1:5], w, h))  # intersection over area
                label = label[ioa < 0.60]  # remove >60% obscured labels

    return image, label


# mixup，把两张图片直接数值上乘以比例再相加
def mixup(image, label, image1, label1):
    if not image.shape == image1.shape:
        # 把图片切割成相同形状（这里不用reshape）
        min_w = min(image.shape[1], image1.shape[1])
        min_h = min(image.shape[0], image1.shape[0])
        image = image[:min_h, :min_w, :]
        image1 = image1[:min_h, :min_w, :]
        # 删除框左上角点位溢出的框
        label = np.delete(label, np.where(label[:, 1] > min_w), axis=0)
        label = np.delete(label, np.where(label[:, 2] > min_h), axis=0)
        label1 = np.delete(label1, np.where(label1[:, 1] > min_w), axis=0)
        label1 = np.delete(label1, np.where(label1[:, 2] > min_h), axis=0)
        # 右下角溢出点位移到边界处
        label[np.where(label[:, 3] > min_w), 3] = min_w
        label[np.where(label[:, 4] > min_h), 4] = min_h
        label1[np.where(label1[:, 3] > min_w), 3] = min_w
        label1[np.where(label1[:, 4] > min_h), 4] = min_h
    # 需要两张图片尺寸相同
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    # r是一个更可能接近0.5的浮点数 有点点类似正态分布
    image0 = (image * r + image1 * (1 - r)).astype(np.uint8)
    label0 = np.concatenate((label, label1), 0)
    return image0, label0


# 垂直翻转
def vertical_flip(image, label):
    image = cv2.flip(image, 0)
    h, w, _ = image.shape
    label[:, 2] = h - label[:, 2]
    label[:, 4] = h - label[:, 4]
    return image, label


# 水平翻转
def horizontal_flip(image, label):
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    label[:, 1] = w - label[:, 1]
    label[:, 3] = w - label[:, 3]
    return image, label


# 水平+垂直翻转
def hv_flip(image, label):
    image = cv2.flip(image, -1)
    h, w, _ = image.shape
    label[:, 1] = w - label[:, 1]
    label[:, 3] = w - label[:, 3]
    label[:, 2] = h - label[:, 2]
    label[:, 4] = h - label[:, 4]
    return image, label


# mosaic4
def load_mosaic4(image_list, label_list, imgsz=640):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4 = []
    mosaic_border = [- imgsz // 2, - imgsz // 2]
    s = imgsz
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
    # print(yc, xc) # yc,xc在320~1280-320，U[320,940]之间均匀分布
    for i in range(4):
        # Load image
        img, (h, w) = image_list[i], image_list[i].shape[:2]

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # x1a, y1a, x2a, y2a 为左上角图的左上角坐标和右下角坐标
            # print("x1a, y1a, x2a, y2a = ",x1a, y1a, x2a, y2a)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            # print("x1b, y1b, x2b, y2b",x1b, y1b, x2b, y2b)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            # print("x1a, y1a, x2a, y2a = ", x1a, y1a, x2a, y2a)
            # print("x1b, y1b, x2b, y2b", x1b, y1b, x2b, y2b)
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            # print("x1a, y1a, x2a, y2a = ", x1a, y1a, x2a, y2a)
            # print("x1b, y1b, x2b, y2b", x1b, y1b, x2b, y2b)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            # print("x1a, y1a, x2a, y2a = ", x1a, y1a, x2a, y2a)
            # print("x1b, y1b, x2b, y2b", x1b, y1b, x2b, y2b)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = (x1a - x1b)
        padh = (y1a - y1b)

        # Labels
        labels = label_list[i].copy()
        # print("i=",i)
        # print("label[i]:",labels)
        # print("w,h,padw,padh = ",w,h,padw,padh)
        labels[:, 1:] = xyxy_2xyxy(labels[:, 1:], padw, padh)  # normalized xywh to pixel xyxy format
        # print("使用坐标转换后的labels",labels)
        labels4.append(labels.copy())
        # print("此时的labels4：",labels4)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:],):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment

    img4, labels4 = random_perspective(img4,
                                       labels4)  # border to remove
    # print("label4.shape:",labels4.shape)
    # print(labels4)
    return img4, labels4


# mosaic9
def load_mosaic9(image_list, label_list, imgsz=640):
    # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
    labels9 = []
    s = imgsz
    mosaic_border = [- imgsz // 2, - imgsz // 2]
    hp, wp = -1, -1  # height, width previous
    for i in range(9):
        # Load image
        img, (h, w) = image_list[i], image_list[i].shape[:2]

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

        # Labels
        labels = label_list[i].copy()
        labels[:, 1:] = xyxy_2xyxy(labels[:, 1:], padx, pady)  # normalized xywh to pixel xyxy format

        labels9.append(labels.copy())

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = (int(random.uniform(0, s)) for _ in mosaic_border)  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers

    for x in (labels9[:, 1:]):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9,
                                       labels9,
                                       border=mosaic_border)  # border to remove

    return img9, labels9


# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    DEMO_model = False  # True为DEMO演示，False为离线增强，demo部分可用于测试参数

    if DEMO_model:
        # 演示模式，演示各图运行效果

        # 载入路径
        image_dir = "demo_dataset/images"
        images = os.listdir(image_dir)
        # 载入图片0和对应label
        image = cv2.imread(image_dir + "/" + images[0])
        label = (image_dir + "/" + images[0]).replace("/images", "/labels").replace(".jpg", ".txt").replace(".png",
                                                                                                            ".txt")
        label = np.loadtxt(label)

        # 载入图片1和对应label (这里额外载入一张图用于图像mix，注意图1要和图0尺寸相同)
        image1 = cv2.imread(image_dir + "/" + images[1])
        label1 = (image_dir + "/" + images[1]).replace("/images", "/labels").replace(".jpg", ".txt").replace(".png",
                                                                                                             ".txt")
        label1 = np.loadtxt(label1)
        # print("label1", label1)
        label1 = xywh_rate2xyxy(image1, label1)
        # print("label:1", label1)
        # 打印原图和对应label
        cv2show(draw_box(image, label), "source image", close=False)
        cv2show(draw_box(image1, label1), "source image1", close=False)


        def debug_module(module_name, func, image, label, **args):
            image_copy = image.copy()
            label_copy = label.copy()
            new_image, new_label = func(image_copy, label_copy, **args)
            print(module_name + "\nnew_image.shape = {}, new_label.shape = {}".format(new_image.shape, new_label.shape))
            cv2show(draw_box(new_image, new_label), module_name, save_image=True)
            print(label_copy)


        # 测试随机HSV、灰度直方
        # hsv_image = augment_hsv(image)
        # cv2show(hsv_image, "HSV image", close=False)
        # out_image = hist_equalize(image, bgr=True)
        # cv2show(out_image, "hist_equalize image", close=False)

        # 测试复制
        debug_module(module_name="replicate image", func=replicate, image=image, label=label)

        # 测试RST
        debug_module(module_name="RST image", func=random_perspective, image=image, label=label)

        # 测试scale
        debug_module(module_name="scale", func=scale, image=image, label=label, imgsz=640)

        # 测试cutout
        debug_module(module_name="cut out", func=cutout, image=image, label=label, p=0.9)

        # 测试mixup (mixup必须两张图尺寸相同)
        new_image, new_label = mixup(image, label, image1, label1)
        debug_module(module_name="mixup", func=mixup, image=image, label=label, image1=image1, label1=label1)

        # 测试上下翻转
        debug_module(module_name="horizontal flip", func=horizontal_flip, image=image, label=label)

        # 测试左右翻转
        debug_module(module_name="vertical flip", func=vertical_flip, image=image, label=label)

        # 测试透视
        debug_module(module_name="random_perspective", func=random_perspective, image=image, label=label, degrees=10,
                     translate=.1,
                     scale=.1, shear=10, perspective=0, border=(0, 0))

        # 测试上下左右翻转hv_flip
        debug_module(module_name="hv_flip", func=hv_flip, image=image, label=label)

        # 测试mosaic4
        image_list = [image, image, image, image]
        label_list = [label, label, label, label]
        img4, labels4 = load_mosaic4(image_list, label_list)
        cv2show(draw_box(img4, labels4), "mosaic,image4")

        # 测试mosaic9
        image_list = [image, image, image, image, image, image, image, image, image]
        label_list = [label, label, label, label, label, label, label, label, label]
        img9, labels9 = load_mosaic9(image_list, label_list, imgsz=2560)
        cv2show(draw_box(img9, labels9), "mosaic,image9")


    else:  #
        # 载入路径
        image_dir = "Dataset/train_dataset/images"  # 这里只对训练数据增强，不对验证集增强
        images = os.listdir(image_dir)
        images_augmentation_dir = image_dir.replace("images", "") + "images_augmentation"
        labels_augmentation_dir = image_dir.replace("images", "") + "labels_augmentation"
        if not os.path.exists(images_augmentation_dir):
            os.makedirs(images_augmentation_dir)
        if not os.path.exists(labels_augmentation_dir):
            os.makedirs(labels_augmentation_dir)


        def apply_module(module_name, func, img_file_name, **args):
            new_image, new_label = func(**args)
            cv2.imwrite(images_augmentation_dir + "/" + img_file_name + "_" + module_name + ".jpg", new_image)
            # np.savetxt(labels_augmentation_dir + "/" + img_file_name + "_" + module_name + ".txt", new_label)
            # 保存时候要保存为浮点数
            h, w, _ = new_image.shape
            save_label = new_label.copy()
            # x
            save_label[:, 1] = np.abs(new_label[:, 1] + new_label[:, 3]) / 2 / w
            # y
            save_label[:, 2] = np.abs(new_label[:, 2] + new_label[:, 4]) / 2 / h
            # w
            save_label[:, 3] = np.abs(new_label[:, 1] - new_label[:, 3])  / w
            # h
            save_label[:, 4] = np.abs(new_label[:, 2] - new_label[:, 4])  / h

            np.savetxt(labels_augmentation_dir + "/" + img_file_name + "_" + module_name + ".txt", save_label,
                       fmt='%.16f')
            print(images_augmentation_dir + "/" + img_file_name + "_" + module_name + ".jpg")
            print(labels_augmentation_dir + "/" + img_file_name + "_" + module_name + ".txt")
        tqdm_bar = tqdm(images)
        import time

        image_list_9 = []
        label_list_9 = []
        for idx, img in enumerate(tqdm_bar):
            start = time.time()
            image = cv2.imread(image_dir + "/" + img)
            read_img_time = time.time()

            image_file_name_no_suffix = img.replace(".jpg", "").replace(".png", "")
            print("image_file_name_no_suffix:", image_file_name_no_suffix)
            label = np.loadtxt((image_dir + "/" + image_file_name_no_suffix).replace("/images", "/labels") + ".txt")
            read_label_time = time.time()

            if len(label.shape) == 1:
                label = np.expand_dims(label, axis=0)
            label = xywh_rate2xyxy(image, label)
            prosess_label_time = time.time()

            # 调用增强
            # 单张图增强
            apply_module("augment_hsv", augment_hsv, image_file_name_no_suffix, image=image, label=label, hgain=0.5,
                         sgain=0.5,
                         vgain=0.5)
            apply_module("hist_equalize", hist_equalize, image_file_name_no_suffix, image=image, label=label,
                         clahe=True, bgr=True)
            apply_module("replicate", replicate, image_file_name_no_suffix, image=image, label=label)
            apply_module("random_perspective", random_perspective, image_file_name_no_suffix, image=image, label=label,
                         degrees=10, translate=.1,
                         scale=.1, shear=10, perspective=0.0, border=(0, 0))
            apply_module("cutout", cutout, image_file_name_no_suffix, image=image, label=label, p=0.5)
            apply_module("vertical_flip", vertical_flip, image_file_name_no_suffix, image=image, label=label)
            apply_module("horizontal_flip", horizontal_flip, image_file_name_no_suffix, image=image, label=label)
            apply_module("hv_flip", hv_flip, image_file_name_no_suffix, image=image, label=label)

            # 多图增强
            # 避免原图尺寸过大，先将原图resize成720尺寸
            image_small, label_small = scale(image, label, imgsz=720)
            image_list_9.append(image_small.copy())
            label_list_9.append(label_small.copy())
            if len(image_list_9) > 9:
                del image_list_9[0]
                del label_list_9[0]

            if idx > 1:
                apply_module("mixup", mixup, image_file_name_no_suffix, image=image_list_9[-2], label=label_list_9[-2],
                             image1=image_list_9[-1], label1=label_list_9[-1])
            if idx > 4:
                apply_module("load_mosaic4", load_mosaic4, image_file_name_no_suffix, image_list=image_list_9[-4:],
                             label_list=label_list_9[-4:], imgsz=1500)

            if idx > 9:
                apply_module("load_mosaic9", load_mosaic9, image_file_name_no_suffix, image_list=image_list_9[-9:],
                             label_list=label_list_9[-9:], imgsz=2500)

            aug_time = time.time()
            # print("读取图像用时:", read_img_time - start)
            # print("读取label用时:", read_label_time - read_img_time)
            # print("label转格式用时:", prosess_label_time - read_label_time)
            # print("增强用时:", aug_time - prosess_label_time)

            # 可以发现离线增强速度主要慢在I/O上
            tqdm_bar.set_description("正在进行图像增强," + "读图:" + str(round((read_img_time - start), 4)) + "s 读标签:" + str(
                round((read_label_time - read_img_time), 4)) + "s 转标签:" + str(
                round((prosess_label_time - read_label_time), 4)) + "s 增强:" + str(
                round((aug_time - prosess_label_time), 4)) + "s ")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import PIL
import sklearn

twopf_1 = cv2.imread('dataset/SLAM_dataset1_0215/Raw image/1/09192016 rat in vivo cancer_1p5cm tumor_stepsize150_4_1080-1140_2PF.png')
threepf_1 = cv2.imread('dataset/SLAM_dataset1_0215/Raw image/1/09192016 rat in vivo cancer_1p5cm tumor_stepsize150_4_1080-1140_3PF.png')
SHG_1 = cv2.imread('dataset/SLAM_dataset1_0215/Raw image/1/09192016 rat in vivo cancer_1p5cm tumor_stepsize150_4_1080-1140_SHG.png')
THG_1 = cv2.imread('dataset/SLAM_dataset1_0215/Raw image/1/09192016 rat in vivo cancer_1p5cm tumor_stepsize150_4_1080-1140_THG.png')
segmask = cv2.imread('dataset/SLAM_dataset1_0215/Segmentation mask/iter65000_c1_09192016 rat in vivo cancer_1p5cm tumor_stepsize150_4_1080-1140.png')

def mask_segment(mask):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    gray_mask[gray_mask < 0.5 * 255] = 0
    gray_mask[gray_mask > 0] = 1

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_mask, connectivity=8)

    return num_labels, labels, stats, centroids

def getRedox(two_pf, three_pf, mask, mask_stats):
    cells = []
    image = (two_pf)/(two_pf + three_pf + 0.01) * mask
    for i, stat in enumerate(mask_stats):
        if stat[4] > 100:
            if stat[4] < 4000:
                'trying to cut the info of each cell off from the big redox image, looks problematic'
                # cells.append(image[stat[0] : stat[0] + stat[2], stat[1] : stat[1] + stat[3]])
    return cells, image

# def redox_segment(redox_image):
#     image = redox_image.astype(np.uint8)
#     binary_redox = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     binary_redox[binary_redox < 0.5 * 255] = 0
#     binary_redox[binary_redox > 0] = 1
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_redox, connectivity=8)
#     return num_labels, labels, stats, centroids

def drawRedoxCell(redox_image, mask_stats):
    realindex = 1
    for i, stat in enumerate(mask_stats):
        if stat[4] > 100:
            if stat[4] < 4000:
                cv2.rectangle(redox_image, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (255, 25, 25), 1)
                cv2.putText(redox_image, str(realindex), (stat[0], stat[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 255, 25), 2)
                realindex += 1


num_labels, labels, stats, centroids = mask_segment(segmask)

redox_cells, redox_image = getRedox(twopf_1, threepf_1, segmask, stats)

# redox_num, redox_labels, redox_stats, redox_ccentroids = redox_segment(redox_image)

color_redox = redox_image.astype(np.uint8)
color = cv2.applyColorMap(color_redox, cv2.COLORMAP_JET)

drawRedoxCell(color, stats)

cv2.imshow('redox',  color)

cv2.waitKey()

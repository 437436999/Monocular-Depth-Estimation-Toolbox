import os
import cv2
import numpy as np
import sys
import shutil
from utils import progress_bar, numerical_sort, calHeightValue, drawBox
import re

img_count = 0
cam_height = 0.0
ERROR_THRESHOLD = 15


gt_height_sum = []
pred_height_sum = []
diff_height_sum = []

# 完成一张rect图片，读取对应的检测结果进行评估
def detectRectImage(gt_file_path, pred_file_path, detect_txt_path):
    global pred_height_sum, gt_height_sum, diff_height_sum
    
    # 读取txt文件
    if not os.path.exists(detect_txt_path):
        print(detect_txt_path,"not exist")
        return
    txt_file = open(detect_txt_path, "r")
    txt_lines = txt_file.readlines()
    if len(txt_lines)==0: # 图像没有检测框，直接返回
        return

    # 读取gt
    if not os.path.exists(gt_file_path):
        print(gt_file_path,"not exist")
        return
    gt_img = cv2.imread(gt_file_path, cv2.IMREAD_ANYDEPTH)
    draw_gt = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
    draw_gt = cv2.convertScaleAbs(draw_gt, alpha=(255.0/65535.0))
    
    # 读取sgbm
    if not os.path.exists(pred_file_path):
        print(pred_file_path,"not exist")
        return
    pred_img = cv2.imread(pred_file_path, cv2.IMREAD_ANYDEPTH)
    draw_pred = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)
    draw_pred = cv2.convertScaleAbs(draw_pred, alpha=(255.0/65535.0))
    
    for line in txt_lines:
        # 检测框的坐标已经在512x512区域
        x, y, xx, yy = map(int, line.strip().split()[:4])
        x1 = int(x)
        x2 = int(xx)
        y1 = int(y)
        y2 = int(yy)

        # 根据检测框位置计算行人身高误差
        pred_value = calHeightValue(x1,x2,y1,y2, pred_img)
        gt_value = calHeightValue(x1,x2,y1,y2, gt_img)

        draw_pred = drawBox(x1, x2, y1, y2, draw_pred, value1=cam_height-pred_value, value2=pred_value)
        draw_gt = drawBox(x1, x2, y1, y2, draw_gt, value1=cam_height-gt_value, value2=gt_value)
        if gt_value!=0:
            pred_height_sum.append(cam_height-pred_value)
            gt_height_sum.append(cam_height-gt_value)
            diff_height_sum.append(abs(pred_value-gt_value))

        if cam_height-gt_value < 150:
            print(gt_file_path)

        
    
    # 保存图片
    file_name = gt_file_path.split("/")[-1][:-4]
    pred_img_save_path = pred_file_path.replace(".png", "_pred_result.png")
    gt_img_save_path = pred_file_path.replace(".png", "_gt_result.png")
    cv2.imwrite(pred_img_save_path, draw_pred)
    cv2.imwrite(gt_img_save_path, draw_gt)

# ======================================================== main ========================================================================================

pred_path = sys.argv[1]         # 预测结果图的保存路径，save_files/2000C_test/
file_list_name = sys.argv[2]    # 图片列表txt文件 data/2000C/2000C_test1.txt
cam_height = int(sys.argv[3])
file_list = open(file_list_name, 'r')
lines_f = file_list.readlines()
lines_f.sort(key=numerical_sort)

# 创建目录用于debug
if os.path.exists(os.path.join(pred_path, "debug")):
    shutil.rmtree(os.path.join(pred_path, "debug"))
os.mkdir(os.path.join(pred_path, "debug"))

# 遍历每一张图片
count = 1
total = len(lines_f)
for line_f in lines_f:
    gt_file_path = line_f.split(" ")[1]
    pred_file_path = os.path.join(pred_path, "data_lab521_" + gt_file_path.replace("/", "_"))
    detect_txt_path = gt_file_path.replace(".png", "_detect.txt")
    # print(pred_file_path, gt_file_path)

    # 输入为rect，对每个rect输入计算评估结果
    detectRectImage(gt_file_path, pred_file_path, detect_txt_path)
    progress_bar(total, count)
    count += 1


# # 输出平均误差
print()
print("detect count = ", len(pred_height_sum))
print("pred result")
print("gt result")
print("differ result")
print("avg, min, max, std = ", np.mean(pred_height_sum), np.min(pred_height_sum), np.max(pred_height_sum), np.std(pred_height_sum))
print("avg, min, max, std = ", np.mean(gt_height_sum), np.min(gt_height_sum), np.max(gt_height_sum), np.std(gt_height_sum))
print("avg, min, max, std = ", np.mean(diff_height_sum), np.min(diff_height_sum), np.max(diff_height_sum), np.std(diff_height_sum))
print(f"'{file_list_name}' metric finish.")

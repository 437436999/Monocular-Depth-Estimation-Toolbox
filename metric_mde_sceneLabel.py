import os
import cv2
import numpy as np
import sys
import shutil
from data.lab521.utils import progress_bar, numerical_sort, calHeightValue, drawBox, string_formatting
from depth.core.evaluation.metrics import metrics
import re

pre_eval_results = []
pre_eval_results_cls = {}
pre_results_cls = {}
gt_results_cls = {}
diff_results = {
    "0-2":[],
    "2-4":[],
    "4-6":[],
    "6-8":[],
    "8-10":[],
}


# 完成一张rect图片，读取对应的检测结果进行评估
def metricOneImage(src_file_path, gt_file_path, pred_file_path, scene_label = "nyu", depth_scale = 1000.0):
    # 读取场景标签
    if scene_label == "sun_cls":
        scene_file = open(os.path.join(gt_file_path[:gt_file_path.rfind("/")].replace("depth", "scene.txt")), "r")
        scene_label = scene_file.readline()
        cls_label = scene_label
    elif scene_label == "sun_cam":
        cls_label = gt_file_path.split("/")[3]
    elif scene_label == "2D3DS":
        cls_label = os.path.basename(src_file_path).split("_")[2]
        area_label = string_formatting(src_file_path).split("/")[2].split("_")[1]
    elif scene_label == "nyu":
        cls_label = gt_file_path.split("/")[-2]
        n = cls_label.find("0")
        if n>0:
            cls_label = cls_label[:n-1] + "_train"
    
    # 读取gt
    if not os.path.exists(gt_file_path):
        print(gt_file_path,"not exist")
        return
    gt_img = cv2.imread(gt_file_path, cv2.IMREAD_ANYDEPTH)
    gt_img = gt_img.astype(np.float32)/depth_scale
    # draw_gt = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
    # draw_gt = cv2.convertScaleAbs(draw_gt, alpha=(255.0/65535.0))
    
    # 读取pred
    if not os.path.exists(pred_file_path):
        print(pred_file_path,"not exist")
        return
    pred_img = cv2.imread(pred_file_path, cv2.IMREAD_ANYDEPTH)
    pred_img = pred_img.astype(np.float32)/6000.0
    
    # 读取原图
    if not os.path.exists(src_file_path):
        print(src_file_path,"not exist")
        return
    src_img = cv2.imread(src_file_path)

    gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # mask
    mask_max = gt_img<10.0
    mask_min = gt_img>1e-3
    mask = np.logical_and(mask_max, mask_min)
    gt = gt_img[mask]
    pred = pred_img[mask]

    # 计算评估指标
    eval = metrics(gt_img, pred_img, min_depth=1e-3, max_depth=10)
    pre_eval_results.append(eval)

    # 统计误差：rmse (0-2m) rmse (2-4m) rmse (4-6m) rmse (6-8m) rmse (8-10m)
    range_begin_list = [0,2,4,6,8]
    for range_begin in range_begin_list:
        range_end = range_begin+2
        mask_range_max = gt<range_end
        mask_range_min = gt>range_begin
        mask_range = np.logical_and(mask_range_max, mask_range_min)
        diff_range = np.abs(gt[mask_range]-pred[mask_range])
        diff_results[f"{range_begin:.0f}-{range_end:.0f}"].append(diff_range.flatten())

    # if not pre_eval_results_cls.__contains__(cls_label):
    #     pre_eval_results_cls[cls_label] = []
    #     pre_results_cls[cls_label] = []
    #     gt_results_cls[cls_label] = []
    # pre_eval_results_cls[cls_label].append(eval)
    # pre_results_cls[cls_label].append(pred_img)
    # gt_results_cls[cls_label].append(gt_img)

    # # 保存图片
    # src_img_save_path = pred_file_path.replace(".png", ".jpg")
    # pred_img_save_path = pred_file_path.replace(".png", "_pred_res.png")
    # gt_img_save_path = pred_file_path.replace(".png", "_gt_res.png")
    # diff_img_save_path = pred_file_path.replace(".png", "_diff_res.png")
    # diff_img = np.abs(gt_img-pred_img)
    # diff_img[gt_img==0] = 0
    # pred_img = (pred_img*6000).astype(np.uint16)
    # gt_img = (gt_img*6000).astype(np.uint16)
    # diff_img = (diff_img*15000).astype(np.uint16)
    # cv2.imwrite(src_img_save_path, src_img)
    # cv2.imwrite(pred_img_save_path, pred_img)
    # cv2.imwrite(gt_img_save_path, gt_img)
    # cv2.imwrite(diff_img_save_path, diff_img)

# ======================================================== main ========================================================================================
# 评估以场景标签为场景信息的预测结果

pred_path = sys.argv[1]         # 预测结果图的保存路径，save_files/2000C_test/
file_list_name = sys.argv[2]    # 图片列表txt文件 data/2000C/2000C_test1.txt
file_list = open(file_list_name, 'r')
lines_f = file_list.readlines()
lines_f.sort(key=numerical_sort)

# # 创建目录用于debug
# if os.path.exists(os.path.join(pred_path, "debug")):
#     shutil.rmtree(os.path.join(pred_path, "debug"))
# os.mkdir(os.path.join(pred_path, "debug"))

# 遍历每一张图片
count = 1
total = len(lines_f)
for line_f in lines_f[:]:
    line_list =  line_f.split(" ")
    line_list[0] = string_formatting(line_list[0])
    line_list[1] = string_formatting(line_list[1])

    sn = file_list_name.split("/")[1]
    src_file_path = os.path.join("data", sn, line_list[0])
    gt_file_path = os.path.join("data", sn, line_list[1])
    pred_file_path = os.path.join(pred_path, f"data_{sn}_" + line_list[0].replace("/", "_").replace("jpg", "png"))

    # 输入为rect，对每个rect输入计算评估结果
    if sn=="nyu":
        metricOneImage(src_file_path, gt_file_path, pred_file_path)
    elif sn=="sunrgbd":
        metricOneImage(src_file_path, gt_file_path, pred_file_path, "sun_cam", 8000.0)
    elif sn=="2D-3D-Semantics":
        metricOneImage(src_file_path, gt_file_path, pred_file_path, "2D3DS", 512.0)

    progress_bar(total, count)
    count += 1

# # 输出平均误差
import matplotlib.pyplot as plt
import pandas as pd
print()
print("metric result")
np.set_printoptions(linewidth=np.inf, precision=4)
print(np.mean(pre_eval_results, axis=0))

# 统计误差：rmse (0-2m) rmse (2-4m) rmse (4-6m) rmse (6-8m) rmse (8-10m)
c = []
r = []
for k,v in diff_results.items():
    v_arrange = np.concatenate(v)
    r.append(np.mean(v_arrange))
    c.append(v_arrange.shape[0])
    # print(k, v_arrange.shape, np.mean(v_arrange))
result_c = ', '.join(map(str, c))
result_r = ', '.join(map(str, r))
print(result_c)
print(result_r)

# 不同场景标签的误差
# for key, value in pre_eval_results_cls.items():
#     key_str = key.ljust(20, " ")
#     print(key_str, "\t", len(value), "\t", np.mean(value, axis=0))

    # 每个分类得到各自的深度值直方图分布

    # # 统计直方图
    # data1_ravel = np.concatenate(pre_results_cls[key])
    # nonzero_indices1 = data1_ravel.nonzero()
    # data1 = data1_ravel[nonzero_indices1]
    # data2_ravel = np.concatenate(gt_results_cls[key])
    # nonzero_indices2 = data2_ravel.nonzero()
    # data2 = data1_ravel[nonzero_indices2]

    # hist1, bins1 = np.histogram(data1, bins=40)  # 直方图统计
    # hist2, bins2 = np.histogram(data2, bins=40) 
    # hist1_norm = hist1.ravel()/hist1.sum()
    # hist2_norm = hist2.ravel()/hist2.sum()
    # bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
    # bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
    # # 绘制折线图
    # plt.plot(bin_centers1, hist1_norm, color='blue', label='gt')
    # plt.plot(bin_centers2, hist2_norm, color='red', label='pred')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title(f'Histogram of scene {key}')
    # plt.legend()# 显示图例
        
    # # 保存图像
    # basename = os.path.basename(file_list_name)[:-4]
    # plt.savefig(os.path.join("histogram", f"{basename}_{key}_histogram.png"))
    # plt.clf()


# # 统计直方图
# data3 = np.array(diff_height_sum)
# hist3, bins3 = np.histogram(data3, bins=20)  # 直方图统计
# bin_centers3 = (bins3[:-1] + bins3[1:]) / 2
# # 绘制折线图
# plt.clf()
# plt.plot(bin_centers3, hist3, color='blue', label='diff')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram Line Plot')
# plt.legend()# 显示图例
# # 保存图像
# plt.savefig(os.path.basename(pred_path[:-1]) + "_" + os.path.basename(file_list_name)[:-4]+'_diff_histogram.png')

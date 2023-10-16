import os
import cv2
import numpy as np
import sys
import shutil
from data.lab521.utils import progress_bar, numerical_sort, calHeightValue, drawBox, cam_heights
from depth.core.evaluation.metrics import metrics
import re
import matplotlib.pyplot as plt
import pandas as pd

img_count = 0
cam_height = 0.0
ERROR_THRESHOLD = 50
GENERATE_HIST = False

gt_height_sum = []
pred_height_sum = []
diff_height_sum = []
gt_height_pre_person = {}
pred_height_pre_person = {}
diff_height_pre_person = {}
pre_eval_results = []

# 完成一张rect图片，读取对应的检测结果进行评估
def detectRectImage(src_file_path, gt_file_path, pred_file_path, detect_txt_path):
    global pred_height_sum, gt_height_sum, diff_height_sum, gt_height_pre_person, pred_height_pre_person, diff_height_pre_person, pre_eval_results
    
    # 读取txt文件
    if not os.path.exists(detect_txt_path):
        print(detect_txt_path,"not exist")
        return
    txt_file = open(detect_txt_path, "r")
    txt_lines = txt_file.readlines()
    if len(txt_lines)==0 or len(txt_lines)>1:
        print(f"'{detect_txt_path}' do not have 1 line, error.")
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
    
    # eval = metrics(gt_img.astype(np.float32)/10000.0, pred_img.astype(np.float32)/10000.0, min_depth=1e-3, max_depth=10)
    # np.set_printoptions(linewidth=np.inf)
    # print("\n", os.path.basename(src_file_path),eval)
    
    # 读取原图
    if not os.path.exists(src_file_path):
        print(src_file_path,"not exist")
        return
    src_img = cv2.imread(src_file_path)
    
    # 检测框的坐标已经在512x512区域
    line = txt_lines[0]
    x1, y1, x2, y2 = map(int, line.strip().split()[4:8])
    track_id = 1
    if len(line.strip().split())==9:
        x1, y1, x2, y2, track_id = map(int, line.strip().split()[4:])

    # 根据检测框位置计算行人身高误差
    pred_value = calHeightValue(x1,x2,y1,y2, pred_img)
    gt_value = calHeightValue(x1,x2,y1,y2, gt_img)

    draw_pred = drawBox(x1, x2, y1, y2, draw_pred, value1=cam_height-pred_value, value2=pred_value)
    draw_gt = drawBox(x1, x2, y1, y2, draw_gt, value1=cam_height-gt_value, value2=gt_value)
    src_img = drawBox(x1, x2, y1, y2, src_img, value1=cam_height-gt_value, value2=cam_height-pred_value)

    depth_scale = 10000.0
    pred_img = pred_img.astype(np.float32)/depth_scale
    gt_img = gt_img.astype(np.float32)/depth_scale
    eval = metrics(gt_img, pred_img, min_depth=1e-3, max_depth=10)
    pre_eval_results.append(eval)


    if gt_value!=0:
        # pred_height_sum.append(cam_height-pred_value)
        # gt_height_sum.append(cam_height-gt_value)
        # diff_height_sum.append(abs(pred_value-gt_value))
        if not gt_height_pre_person.__contains__(track_id):
            gt_height_pre_person[track_id] = []
            pred_height_pre_person[track_id] = []
            diff_height_pre_person[track_id] = []
        gt_height_pre_person[track_id].append(cam_height-gt_value)
        pred_height_pre_person[track_id].append(cam_height-pred_value)
        diff_height_pre_person[track_id].append(abs(pred_value-gt_value))

    # 保存图片
    file_name = gt_file_path.split("/")[-1][:-4]
    src_img_save_path = pred_file_path.replace(".png", ".jpg")
    pred_img_save_path = pred_file_path.replace(".png", "_pred_res.png")
    gt_img_save_path = pred_file_path.replace(".png", "_gt_res.png")
    cv2.imwrite(src_img_save_path, src_img)
    cv2.imwrite(pred_img_save_path, draw_pred)
    cv2.imwrite(gt_img_save_path, draw_gt)

    track_path = os.path.join(pred_path, str(track_id))
    if not os.path.exists(track_path):
        os.mkdir(track_path)
    shutil.copy(pred_img_save_path, track_path)
    shutil.copy(src_img_save_path, track_path)

    if abs(pred_value-gt_value) > ERROR_THRESHOLD:
        shutil.copy(pred_file_path, debug_path)
        shutil.move(src_img_save_path, debug_path)
        shutil.move(pred_img_save_path, debug_path)
        shutil.move(gt_img_save_path, debug_path)
        return

# ======================================================== main ========================================================================================
# 评估以相机挂高为场景信息的预测结果

pred_path = sys.argv[1]         # 预测结果图的保存路径，save_files/2000C_test/
file_list_name = sys.argv[2]    # 图片列表txt文件 data/2000C/2000C_test1.txt
cam_height = 0
if len(sys.argv)<4:
    txt_file = os.path.basename(file_list_name)
    first_underscore_index = txt_file.index('_')
    try:
        second_underscore_index = txt_file.index('_', first_underscore_index + 1)
    except ValueError:
        second_underscore_index = -4 # 截取到.txt
    cam_key = txt_file[:second_underscore_index]
    if cam_heights.__contains__(cam_key):
        cam_height = cam_heights[cam_key]
else:
    cam_height = int(sys.argv[3])
file_list = open(file_list_name, 'r')
lines_f = file_list.readlines()
lines_f.sort(key=numerical_sort)

# 创建目录用于debug
if os.path.exists(os.path.join(pred_path, "debug")):
    shutil.rmtree(os.path.join(pred_path, "debug"))
os.mkdir(os.path.join(pred_path, "debug"))
debug_path = os.path.join(pred_path, "debug")

# 遍历每一张图片
count = 1
total = len(lines_f)
for line_f in lines_f:
    dataset_name = os.path.basename(file_list_name).split("_")[0]
    src_file_path = os.path.join("data", dataset_name, line_f.split(" ")[0])
    gt_file_path = os.path.join("data", dataset_name, line_f.split(" ")[1])
    pred_file_path = os.path.join(pred_path, src_file_path.replace("/", "_").replace("jpg", "png"))
    detect_txt_path = gt_file_path.replace(".png", ".txt")
    # print(pred_file_path, gt_file_path)

    # 输入为rect，对每个rect输入计算评估结果
    detectRectImage(src_file_path, gt_file_path, pred_file_path, detect_txt_path)
    progress_bar(total, count)
    count += 1

# 每个行人的平均误差
pred_mean = []
pred_std = []
gt_mean = []
gt_std = []
diff_mean = []
diff_std = []
data = []
for track_id, v in gt_height_pre_person.items():
    # print("track id:", track_id, "num = ", len(gt_height_pre_person[track_id]))
    # print(np.mean(pred_height_pre_person[track_id]), np.min(pred_height_pre_person[track_id]), np.max(pred_height_pre_person[track_id]), np.std(pred_height_pre_person[track_id]))
    # print(np.mean(gt_height_pre_person[track_id]), np.min(gt_height_pre_person[track_id]), np.max(gt_height_pre_person[track_id]), np.std(gt_height_pre_person[track_id]))
    # print(np.mean(diff_height_pre_person[track_id]), np.min(diff_height_pre_person[track_id]), np.max(diff_height_pre_person[track_id]), np.std(diff_height_pre_person[track_id]))
    track_data = {
        "Track ID": track_id,
        "Track num": len(gt_height_pre_person[track_id]),
        "GT Mean": np.mean(gt_height_pre_person[track_id]),
        "GT Min": np.min(gt_height_pre_person[track_id]),
        "GT Max": np.max(gt_height_pre_person[track_id]),
        "GT Std": np.std(gt_height_pre_person[track_id]),
        "Pred Mean": np.mean(pred_height_pre_person[track_id]),
        "Pred Min": np.min(pred_height_pre_person[track_id]),
        "Pred Max": np.max(pred_height_pre_person[track_id]),
        "Pred Std": np.std(pred_height_pre_person[track_id]),
        "Diff Mean": np.mean(diff_height_pre_person[track_id]),
        "Diff Min": np.min(diff_height_pre_person[track_id]),
        "Diff Max": np.max(diff_height_pre_person[track_id]),
        "Diff Std": np.std(diff_height_pre_person[track_id]),
    }
    data.append(track_data)

    if track_id==-1 or np.std(gt_height_pre_person[track_id])<100:
        pred_mean.append(np.mean(pred_height_pre_person[track_id]))
        pred_std.append(np.std(pred_height_pre_person[track_id]))
        pred_height_sum.extend(pred_height_pre_person[track_id])
        gt_mean.append(np.mean(gt_height_pre_person[track_id]))
        gt_std.append(np.std(gt_height_pre_person[track_id]))
        gt_height_sum.extend(gt_height_pre_person[track_id])
        diff_mean.append(np.mean(diff_height_pre_person[track_id]))
        diff_std.append(np.std(diff_height_pre_person[track_id]))
        diff_height_sum.extend(diff_height_pre_person[track_id])

df = pd.DataFrame(data)
df.to_excel("excel/" + os.path.basename(pred_path[:-1]) + "_" + os.path.basename(file_list_name)[:-4] + '.xlsx', index=False)

# # 输出平均误差
# print()
# print("detect count = ", len(pred_height_sum))
np.set_printoptions(linewidth=np.inf)
print("\r", ' '.join(str(item) for item in np.mean(pre_eval_results, axis=0)[:-2]), f"{np.mean(diff_mean):.4f} {np.min(diff_height_sum):.4f} {np.max(diff_height_sum):.4f} {np.mean(diff_std):.4f}", flush=True)
# print("pred\t", np.mean(pred_mean), np.min(pred_height_sum), np.max(pred_height_sum), np.mean(pred_std))
# print("gt\t", np.mean(gt_mean), np.min(gt_height_sum), np.max(gt_height_sum), np.mean(gt_std))
# print("diff\t", np.mean(diff_mean), np.min(diff_height_sum), np.max(diff_height_sum), np.mean(diff_std))
# print(f"'{file_list_name}' metric finish.")

if GENERATE_HIST:
    # 统计直方图
    data1 = cam_height-np.array(gt_height_sum)
    data2 = cam_height-np.array(pred_height_sum)
    # df = pd.DataFrame({'Gt': data1, 'Pred': data2})
    # excel_file = 'data.xlsx'
    # df.to_excel(excel_file, index=False)

    # 统计直方图
    hist1, bins1 = np.histogram(data1, bins=40)  # 直方图统计
    hist2, bins2 = np.histogram(data2, bins=40) 
    bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
    bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
    # 绘制折线图
    plt.clf()
    plt.plot(bin_centers1, hist1, color='blue', label='gt')
    plt.plot(bin_centers2, hist2, color='red', label='pred')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Line Plot')
    plt.legend()# 显示图例
    # 保存图像
    plt.savefig("histogram/" + os.path.basename(pred_path[:-1]) + "_" + os.path.basename(file_list_name)[:-4]+'_histogram.png')


    # 统计直方图
    data3 = np.array(diff_height_sum)
    hist3, bins3 = np.histogram(data3, bins=20)  # 直方图统计
    bin_centers3 = (bins3[:-1] + bins3[1:]) / 2
    # 绘制折线图
    plt.clf()
    plt.plot(bin_centers3, hist3, color='blue', label='diff')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Line Plot')
    plt.legend()# 显示图例
    # 保存图像
    plt.savefig("histogram/" + os.path.basename(pred_path[:-1]) + "_" + os.path.basename(file_list_name)[:-4]+'_diff_histogram.png')

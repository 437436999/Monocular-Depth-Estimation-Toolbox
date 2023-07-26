import json
import matplotlib.pyplot as plt
import sys
import os

# 读取 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        next(f)  # 跳过第一行
        data = [json.loads(line) for line in f]
    return data

# 获取损失数据
def get_loss_data(data, key):
    return [entry[key] for entry in data]

# 绘制折线图
# 绘制折线图
def plot_line_chart(data, file_path):
    epochs = []
    iters = []
    aux_loss_depth_2 = []
    aux_loss_depth_5 = []
    loss_ce = []
    loss_depth = []
    loss = []
    grad_norm = []

    epochs_val = []
    iters_val = []
    a1_val = []
    a2_val = []
    a3_val = []
    abs_rel_val = []
    rmse_val = []
    log_10_val = []
    rmse_log_val = []

    for entry in data:
        if entry['mode'] == "val":
            epochs_val.append(entry['epoch'])
            iters_val.append(cur_iter)
            a1_val.append(entry['a1'])
            a2_val.append(entry['a2'])
            a3_val.append(entry['a3'])
            abs_rel_val.append(entry['abs_rel'])
            rmse_val.append(entry['rmse'])
            log_10_val.append(entry['log_10'])
            rmse_log_val.append(entry['rmse_log'])
            continue

        cur_iter = entry['iter']
        epochs.append(entry['epoch'])
        iters.append(entry['iter'])
        aux_loss_depth_2.append(entry['decode.aux_loss_depth_2'])
        aux_loss_depth_5.append(entry['decode.aux_loss_depth_5'])
        loss_depth.append(entry['decode.loss_depth'])
        loss.append(entry['loss'])
        if 'decode.loss_ce' in entry:
            loss_ce.append(entry['decode.loss_ce'])
        # grad_norm.append(entry['grad_norm'])

    fig, ax1 = plt.subplots()
    ax1.plot(iters, aux_loss_depth_2, label='aux_loss_depth_2')
    ax1.plot(iters, aux_loss_depth_5, label='aux_loss_depth_5')
    ax1.plot(iters, loss_depth, label='loss_depth')
    ax1.plot(iters, loss, label='loss')
    if len(loss_ce)>0:
        ax1.plot(iters, loss_ce, label='loss_ce')
    # ax1.plot(iters, grad_norm, label='grad_norm')

    ax1.set_xlabel('iter')
    ax1.set_ylabel('Loss')
    plt.legend()

    ax2 = ax1.twiny()
    ax2.plot(epochs, aux_loss_depth_2, alpha=0)  # 使用透明的线条来显示 epoch 轴
    ax2.set_xlabel('epoch')

    fig.set_size_inches(10, 8)  # 设置图片大小为 10x6 英寸
    save_path = os.path.join(os.path.dirname(file_path), "chartTrain.png")
    plt.savefig(save_path)

    
    fig, ax3 = plt.subplots()
    ax3.plot(iters_val, a1_val, label='a1')
    ax3.plot(iters_val, a2_val, label='a2')
    ax3.plot(iters_val, a3_val, label='a3')
    ax3.plot(iters_val, abs_rel_val, label='abs_rel')
    ax3.plot(iters_val, rmse_val, label='rmse')
    ax3.plot(iters_val, log_10_val, label='log_10')
    ax3.plot(iters_val, rmse_log_val, label='rmse_log')

    ax3.set_xlabel('iter')
    ax3.set_ylabel('Loss')
    plt.legend()

    ax4 = ax3.twiny()
    ax4.plot(epochs_val, a1_val, alpha=0)  # 使用透明的线条来显示 epoch 轴
    ax4.set_xlabel('epoch')

    # plt.legend()
    fig.set_size_inches(10, 8)  # 设置图片大小为 10x6 英寸
    save_path = os.path.join(os.path.dirname(file_path), "chartVal.png")
    plt.savefig(save_path)


# 主函数
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <json_file_path>")
        return
    
    file_path = sys.argv[1]  # 从命令行参数获取 JSON 文件路径
    data = read_json_file(file_path)
    plot_line_chart(data, file_path)

if __name__ == '__main__':
    main()

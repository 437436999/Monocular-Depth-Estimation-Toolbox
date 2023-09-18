#!/bin/bash
python tools/train.py configs/binsformer/binsformer_swint_w7_2000C_$1.py
python tools/test.py work_dirs/binsformer_swint_w7_2000C_$1/binsformer_swint_w7_2000C_$1.py work_dirs/binsformer_swint_w7_2000C_$1/latest.pth --show-dir save_files/2000C_$1/ --format-only --dataset_root data/2000C/ --dataset_split txt/2000C_$1.txt
python tools/test.py work_dirs/binsformer_swint_w7_2000C_$1/binsformer_swint_w7_2000C_$1.py work_dirs/binsformer_swint_w7_2000C_$1/latest.pth --eval x --dataset_root data/2000C/ --dataset_split txt/2000C_$1.txt
# python metric_mde_v2.py save_files/2000C_$1/ data/2000C/txt/2000C_$1_train.txt
python metric_mde_v2.py save_files/2000C_$1/ data/2000C/txt/2000C_$1_test.txt
 



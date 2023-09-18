# !/bin/bash
# python tools/test.py work_dirs/binsformer_swint_w7_2000C_all/binsformer_swint_w7_2000C_all.py work_dirs/binsformer_swint_w7_2000C_all/latest.pth --show-dir save_files/2000C_all/ --format-only --dataset_root data/2000C/ --dataset_split txt/2000C_all_test.txt

# names="a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a15"
# for name in $names
# do
#     # python tools/test.py work_dirs/binsformer_swint_w7_2000C_all/binsformer_swint_w7_2000C_all.py work_dirs/binsformer_swint_w7_2000C_all/latest.pth --show-dir save_files/2000C_all/ --format-only --dataset_root data/2000C/ --dataset_split txt/2000C_${name}_test.txt
#     python tools/test.py work_dirs/binsformer_swint_w7_2000C_all/binsformer_swint_w7_2000C_all.py work_dirs/binsformer_swint_w7_2000C_all/latest.pth --eval x --dataset_root data/2000C/ --dataset_split txt/2000C_${name}_test.txt
#     # mv cls_res.txt cls_res_${name}_cls_s.txt
#     python metric_mde_v2.py save_files/2000C_all/ data/2000C/txt/2000C_${name}_test.txt
# done

names="110_bj 110_gb 110_zy 120_bj 120_gb 120_zy"
for name in $names
do
    python tools/test.py work_dirs/binsformer_swint_w7_2000C_all/binsformer_swint_w7_2000C_all.py work_dirs/binsformer_swint_w7_2000C_all/latest.pth --show-dir save_files/2000C_all/ --format-only --dataset_root data/lab521/ --dataset_split txt/lab521_${name}_0816_test.txt
    python tools/test.py work_dirs/binsformer_swint_w7_2000C_all/binsformer_swint_w7_2000C_all.py work_dirs/binsformer_swint_w7_2000C_all/latest.pth --eval x --dataset_root data/lab521/ --dataset_split txt/lab521_${name}_0816_test.txt
    python metric_mde_v2.py save_files/2000C_all/ data/lab521/txt/lab521_${name}_0816_test.txt
    # mv cls_res.txt cls_res_${name}_cls_s.txt
done

# !/bin/bash
# python -W ignore tools/test.py work_dirs/binsformerVer5_swint_w7_2000C/binsformerVer5_swint_w7_2000C.py work_dirs/binsformerVer5_swint_w7_2000C/latest.pth --show-dir save_files/binsformerVer5/2000C/ --format-only --dataset_root data/2000C/ --dataset_split txt/2000C_all_test.txt

# names="a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a15"
# for name in $names
# do
#     mkdir save_files/binsformerVer5/2000C/${name}
#     mv save_files/binsformerVer5/2000C/*_${name}_* save_files/binsformerVer5/2000C/${name}
#     # python -W ignore tools/test.py work_dirs/binsformerVer5_swint_w7_2000C/binsformerVer5_swint_w7_2000C.py work_dirs/binsformerVer5_swint_w7_2000C/latest.pth --dataset_root data/2000C/ --dataset_split txt/2000C_${name}_test.txt --eval x
# done

# for name in $names
# do
#     python metric_mde_cameraHeight.py save_files/binsformerVer5/2000C/${name}/ data/2000C/txt/2000C_${name}_test.txt
# done


names="a11 a12 a13 a14 a16"
for name in $names
do
    mkdir save_files/binsformerVer2/2000C_all/${name}
    python -W ignore tools/test.py work_dirs/binsformerVer2_swint_w7/binsformerVer2_swint_w7_2000C_all/binsformerVer2_swint_w7_2000C.py work_dirs/binsformerVer2_swint_w7/binsformerVer2_swint_w7_2000C_all/latest.pth --dataset_root data/2000C/ --dataset_split txt/2000C_${name}.txt --show-dir save_files/binsformerVer2/2000C_all/${name} --format-only
done

for name in $names
do
    python metric_mde_cameraHeight.py save_files/binsformerVer2/2000C_all/${name}/ data/2000C/txt/2000C_${name}.txt
done

# names="110_bj 110_gb 110_zy 120_bj 120_gb 120_zy"
# for name in $names
# do
#     python -W ignore tools/test.py work_dirs/binsformerVer5_swint_w7_2000C/binsformerVer5_swint_w7_2000C.py work_dirs/binsformerVer5_swint_w7_2000C/latest.pth --show-dir save_files/2000C_all/ --format-only --dataset_root data/lab521/ --dataset_split txt/lab521_${name}_0816_test.txt
#     # python -W ignore tools/test.py work_dirs/binsformerVer5_swint_w7_2000C/binsformerVer5_swint_w7_2000C.py work_dirs/binsformerVer5_swint_w7_2000C/latest.pth --eval x --dataset_root data/lab521/ --dataset_split txt/lab521_${name}_0816_test.txt
#     python metric_mde_cameraHeight.py save_files/2000C_all/ data/lab521/txt/lab521_${name}_0816_test.txt
#     # mv cls_res.txt cls_res_${name}_cls_s.txt
# done

# 单数据集
# python -W ignore tools/train.py configs/binsformerVer3/binsformerVer3_swint_w7_2000C.py --work-dir work_dirs/binsformerVer2_swint_w7_2000C_$1/  --dataset_root data/2000C/ --train_dataset_split txt/2000C_$1_train.txt --test_dataset_split txt/2000C_$1_test.txt
# python -W ignore tools/test.py work_dirs/binsformerVer2_swint_w7_2000C_$1/binsformerVer3_swint_w7_2000C.py work_dirs/binsformerVer2_swint_w7_2000C_$1/latest.pth --show-dir save_files/binsformerVer3/2000C_$1/ --format-only --dataset_root data/2000C/ --dataset_split txt/2000C_$1.txt
# python metric_mde_cameraHeight.py save_files/binsformerVer3/2000C_$1/ data/2000C/txt/2000C_$1_test.txt

# 混合数据集
# python -W ignore tools/train.py configs/binsformerVer3/binsformerVer3_swint_w7_2000C.py --work-dir work_dirs/binsformerVer3_swint_w7_2000C_all/ --dataset_root data/2000C/ --train_dataset_split txt/2000C_all_train.txt --test_dataset_split txt/2000C_all_test.txt
# python tools/test.py configs/binsformerVer3/binsformerVer3_swint_w7_2000C.py work_dirs/binsformerVer3_swint_w7_2000C_all/latest.pth --show-dir save_files/binsformerVer3/2000C_all/ --format-only --dataset_root data/2000C/ --dataset_split txt/2000C_all_test.txt

names="a11 a12 a13 a14 a16"
for name in $names
do
    python -W ignore tools/test.py configs/binsformerVer3/binsformerVer3_swint_w7_2000C.py work_dirs/binsformerVer3_swint_w7_2000C_all/latest.pth --show-dir save_files/binsformerVer3/2000C_all/${name}/ --format-only --dataset_root data/2000C/ --dataset_split txt/2000C_${name}_test.txt
done
for name in $names
do
    python metric_mde_cameraHeight.py save_files/binsformerVer3/2000C_all/${name}/ data/2000C/txt/2000C_${name}_test.txt
done

names="110_bj 110_gb 110_zy 120_bj 120_gb 120_zy"
for name in $names
do
    python -W ignore tools/test.py configs/binsformerVer3/binsformerVer3_swint_w7_2000C.py work_dirs/binsformerVer3_swint_w7_2000C_all/latest.pth --show-dir save_files/binsformerVer3/2000C_all/${name}/ --format-only --dataset_root data/lab521/ --dataset_split txt/lab521_${name}_0816_test.txt
done

names="110_bj 110_gb 110_zy 120_bj 120_gb 120_zy"
for name in $names
do
    python metric_mde_cameraHeight.py save_files/binsformerVer3/2000C_all/${name}/ data/lab521/txt/lab521_${name}_0816_test.txt
done




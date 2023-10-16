
names="a9"
for name in $names
do
    python -W ignore tools/test.py configs/binsformerVer2/binsformerVer2_swint_w7_2000C.py work_dirs/binsformerVer2_swint_w7_2000C_all/latest.pth --show-dir save_files/binsformerVer2/2000C_all\(LooseThreshold\)/${name}/ --format-only --dataset_root data/2000C/ --dataset_split txt/2000C_${name}\(LooseThreshold\)_test.txt
done
names="M_area_1_i50_office"
for name in $names
do
    python -W ignore tools/test.py configs/binsformerVer4/binsformerVer4_swint_w7_nyu.py work_dirs/binsformerVer4_swint_w7_nyu/latest.pth --dataset_root data/2D-3D-Semantics/ --dataset_split txt/${name}.txt --batch_size 4 --eval x
    # python -W ignore tools/test.py configs/binsformer/binsformer_swint_w7_nyu.py work_dirs/binsformer_swint_w7_nyu/latest.pth --dataset_root data/2D-3D-Semantics/ --dataset_split txt/${name}.txt --batch_size 4 --eval x
done

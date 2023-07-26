#!/bin/bash
names="2000C_S_330"
# names="2000C_G_247 2000C_W_810 2000C_K_154 2000C_B_all 2000C_I_523 2000C_H_130 2000C_D_041 2000C_A_112 2000C_W_546 2000C_W_373 2000C_U_174 2000C_M_223 2000C_N_187 2000C_S_330 2000C_V_247"

for name in $names
do
    # python tools/test.py work_dirs/binsformer_swint_w7_${name}/binsformer_swint_w7_${name}.py work_dirs/binsformer_swint_w7_${name}/latest.pth --show-dir save_files/${name} --format-only
    # python tools/test.py work_dirs/binsformer_swint_w7_${name}/binsformer_swint_w7_${name}.py work_dirs/binsformer_swint_w7_${name}/latest.pth --show-dir save_files/${name}_train --format-only
    pass
done

for name in $names
do
    python ../metric_tool/metric_2000C_rect.py bins save_files/${name}/ ../../../dataset/2000C/${name}_rect_70_100_test.txt rect2rect
    # python ../metric_tool/metric_2000C_rect.py bins save_files/${name}_train/ ../../../dataset/2000C/${name}_rect_0_70_train.txt rect2rect
    pass
done
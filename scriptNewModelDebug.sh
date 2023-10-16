#!/bin/bash
# python -W ignore tools/train.py configs/binsformerVer2/binsformerVer2_swint_w7_2000C.py --work-dir work_dirs/binsformerVer2_swint_w7_2000C_all/
python -W ignore tools/test.py configs/binsformerVer2/binsformerVer2_swint_w7_2000C.py work_dirs/binsformerVer2_swint_w7_2000C_all/latest.pth --eval x



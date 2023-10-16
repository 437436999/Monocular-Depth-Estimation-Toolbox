# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pdb

from mmcv.utils import print_log
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS
from depth.my_utils import cam_heights

from .nyu import NYUDataset

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

@DATASETS.register_module()
class DataWithCameraHeight(NYUDataset):
    """NYU dataset for depth estimation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── NYU
        │   │   ├── nyu_train.txt
        │   │   ├── nuy_test.txt
        │   │   ├── scenes_xxxx (xxxx. No. of the scenes)
        │   │   │   ├── data_1
        │   │   │   ├── data_2
        │   │   │   |   ...
        │   │   │   |   ...
        |   │   ├── scenes (test set, no scene No.)
        │   │   │   ├── data_1 ...
    split file format:
    input_image: /kitchen_0028b/rgb_00045.jpg
    gt_depth:    /kitchen_0028b/sync_depth_00045.png
    focal:       518.8579
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.png'
        ann_dir (str, optional): Path to annotation directory. Default: None
        depth_map_suffix (str): Suffix of depth maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
    """
 
    def __init__(self,
                 pipeline,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 depth_scale=1000,
                 garg_crop=False,
                 eigen_crop=True,
                 min_depth=1e-3,
                 max_depth=10):

        super(DataWithCameraHeight, self).__init__(pipeline, 
                                                  split,
                                                  data_root,
                                                  test_mode,
                                                  depth_scale,
                                                  garg_crop,
                                                  eigen_crop,
                                                  min_depth,
                                                  max_depth)

    def load_annotations(self, data_root, split):
        """Load annotation from directory.
        Args:
            data_root (str): Data root for img_dir/ann_dir.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        self.invalid_depth_num = 0
        img_infos = []
        class_dict = {}
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == 'None':
                        self.invalid_depth_num += 1
                        continue
                    img_info['ann'] = dict(depth_map=osp.join(data_root, remove_leading_slash(depth_map)))
                    img_name = line.strip().split(" ")[0]
                    img_info['filename'] = osp.join(data_root, remove_leading_slash(img_name))
                    img_infos.append(img_info)

                    # if self.test_mode is not True:

                    # 根据不同前缀进行高度分类
                    if data_root == 'data/2000C/':      # img_name.split("/") ==== ['a10_202222010412_337.34_441', 'rect', 'frame_431_sgbm_113.jpg']
                        # 以场景分类
                        cls_name = img_name.split("/")[0].split("_")[0]
                        # 以高度分类
                        cls_name = cam_heights["2000C_" + cls_name]

                    if data_root == 'data/lab521/':      # ['110_bj2_0816', 'rect', 'frame_1039_sgbm_3.jpg']
                        # 以场景分类
                        cls_name = img_name.split("/")[0].split("_")[0]
                        # 以高度分类
                        cls_name = cam_heights["lab521_" + cls_name]
                        
                    if cls_name not in class_dict.keys():
                        class_dict[cls_name] = len(class_dict.keys())
                        print("new class:", cls_name, class_dict[cls_name])
                    label = class_dict[cls_name]
                    img_info['ann']['class_label'] = label # from 0 - 248 (totally 249 classes)
                    img_info['camera_height'] = cls_name/100.0 # 相机高度，以m为单位

        else:
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images. Totally {self.invalid_depth_num} invalid pairs are filtered', logger=get_root_logger())
        return img_infos
    
    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['depth_scale'] = self.depth_scale
        results['camera_height'] = results['img_info']['camera_height']
        

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)

        # add class label
        results['class_label'] = results['ann_info']['class_label']
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)

        return self.pipeline(results)

    

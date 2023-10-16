# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import json

from mmcv.utils import print_log
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS
from depth.my_utils import S2D3D_scene_info, S2D3D_area_scene_info

from .nyu import NYUDataset


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

@DATASETS.register_module()
class Sem2D3DDataset(NYUDataset):
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
                 depth_scale=512,
                 garg_crop=False,
                 eigen_crop=True,
                 min_depth=1e-3,
                 max_depth=10):

        super(Sem2D3DDataset, self).__init__(pipeline, 
                                                  split,
                                                  data_root,
                                                  test_mode,
                                                  depth_scale,
                                                  garg_crop,
                                                  eigen_crop,
                                                  min_depth,
                                                  max_depth)
        
        self.depth_scale = 512.0
        self.eigen_crop = False


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

                    # 以场景分类
                    cls_name = osp.basename(img_name).split("_")[2]
                    area_name = remove_leading_slash(img_name).split("/")[0].split("_")[1]
                    
                    if cls_name not in class_dict.keys():
                        class_dict[cls_name] = len(class_dict.keys())
                        print("new class:", cls_name, class_dict[cls_name])
                    
                    label = class_dict[cls_name]
                    img_info['ann']['class_label'] = label # from 0 - 248 (totally 249 classes)
                    img_info['scene_info'] = S2D3D_area_scene_info[f"area{area_name}_{cls_name}"]
                    # 读取相机内参  
                    pose_file_name = os.path.join(self.data_root, img_name.replace("rgb", "pose").replace(".png", ".json"))
                    with open(pose_file_name, 'r') as json_file:
                        data = json.load(json_file)
                    camera_k_matrix = data.get("camera_k_matrix", [])
                    if isinstance(camera_k_matrix, list) and len(camera_k_matrix) == 3 and len(camera_k_matrix[0]) == 3:
                        focal_scale_x = camera_k_matrix[0][0] / 518.857901
                        focal_scale_y = camera_k_matrix[1][1] / 519.469611
                        img_info['focal_scale'] = (focal_scale_x, focal_scale_y)
                    else:
                        print("无法提取焦距信息，数据格式不符合预期。")
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
        results['scene_info'] = results['img_info']['scene_info']
        results['focal_scale'] = results['img_info']['focal_scale']

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

    

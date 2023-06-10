# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmcv.utils import print_log
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS

from .nyu import NYUDataset


cam_heights = {'202222010329': 247,
        '202222010027': 273,
        '202222010028': 273,
        '202222010018': 274,
        '202222010003': 280,
        'HUKO_staff_202222010003': 280,
        '202222010333': 280,
        '202222010082': 300,
        '202222010347': 300,
        '202222020174': 300,
        '202222010070': 300,
        '202222020531': 300,
        '202222010257': 306,
        '202222010220': 310,
        '202222010161': 310,
        '202222010135': 320,
        '202222010412': 324,
        '202222010187': 325,
        '202222010235': 333,
        '202222010130': 342,
        '202222010041': 350,
        '202222010247': 350,
        '202222010330': 350,
        '202222020247': 350,
        '202222010169': 350,
        '202222010232': 350,
        '202222010256': 350,
        '202222010350': 350,
        '202222020303': 350,
        '202222020311': 350,
        '202222020449': 350,
        '202222020741': 350,
        '202222020810': 350,
        '202222010152': 360,
        '202222010238': 360,
        '202222020316': 360,
        '202222020373': 360,
        '202222020631': 360,
        '202222010112': 370,
        '202222010065': 380,
        '202222010090': 380,
        '202222010253': 380,
        '202222010339': 380,
        '202222010345': 385,
        '202222010464': 385,
        '202222010341': 400,
        '202222010008': 420,
        '202222010378': 440,
        '202222010029': 274,
        '202222010022': 275,
        '202222010338': 280,
        '202222020306': 280,
        '202222010035': 320,
        '202222020523': 325,
        '202222010154': 344,
        '202222010404': 350,
        '202222020546': 350,
        '202222020767': 360,
        '202222010223': 386,
        '202222010369': 400
        }


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

@DATASETS.register_module()
class NYUBinFormerDataset(NYUDataset):
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

        super(NYUBinFormerDataset, self).__init__(pipeline, 
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

                    if self.test_mode is not True:
                        # 以场景分类
                        cls_name = img_name.split("/")[-1].split("_")[0] + "_" + img_name.split("/")[-1].split("_")[1]
                        # 以高度分类
                        # cls_name = img_name.split("/")[-1].split("_")[1]
                        # if cls_name=="R":
                        #     cls_name = str(cam_heights[img_name.split("/")[-1].split("_")[0]])

                        # NYU原分类代码
                        cls_name = img_name.split("/")[1].split("_")[0]
                        # cls_name = img_name.split("/")[1]
                        # if cls_name[-1].isalpha():
                        #     cls_name = cls_name[:-1]
                            
                        if cls_name not in class_dict.keys():
                            class_dict[cls_name] = len(class_dict.keys())
                            print("new class:", cls_name, class_dict[cls_name])
                        label = class_dict[cls_name]
                        img_info['ann']['class_label'] = label # from 0 - 248 (totally 249 classes)

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

    

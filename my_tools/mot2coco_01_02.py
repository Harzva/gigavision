import pandas as pd
import json
import cv2
from tqdm import tqdm
import numpy as np
import os
PANDA_VIDEO_SEQS = [
    '',
    '01_University_Canteen',
    '02_OCT_Habour',
    '03_Xili_Crossroad',
    '04_Primary_School',
    '05_Basketball_Court',
    '06_Xinzhongguan',
    '07_University_Campus',
    '08_Xili_Street_1',
    '09_Xili_Street_2',

    '10_Huaqiangbei',
    '11_Train_Station_Square',
    '12_Nanshan_i_Park',
    '13_University_Playground',
    '14_Ceremony',
    '15_Dongmen_Street'
]
# PANDA_VIDEO_SEQS = [
#     '02_OCT_Habour',
#     '04_Primary_School',
#     '06_Xinzhongguan',
#     '08_Xili_Street_1',
#     '09_Xili_Street_2',
# ]
class PANDA2COCO:
    '''
    transfer panda annotations into coco annotations
    '''
    def __init__(self,root='/root/data/gvision/dataset/mot/train',mode='train'):
        self.mode=mode
        self.root=root                    
        self.categories = []    # 存储categories键对应的数据
        self.img_id = 0         # 统计image的id
        self.ann_id = 0         # 统计annotation的id 
        self.annotations=[]
        self.images=[]
    
    def _categories(self, num_categories=1):   # num_categories 为总的类别数
        for i in range(0, num_categories):
            category = {}
            category['id'] = i+1
            category['name'] = 'None'             # 可根据实际需要修改
            category['supercategory'] = 'full body'    # 可根据实际需要修改
            self.categories.append(category)
    
    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image
 
    def _annotation(self, bbox,label=1):
        bbox = bbox.tolist()
        area = bbox[2] * bbox[3]
        points = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] =2# label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = bbox
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation
    
    def to_coco(self,dataset_dir):
        """
        anno_file: 自己数据的文件路径
        img_dir: 图片文件夹路径（coco分为train和valid）
        num_categories: bbox对应的总类别数目
        """
        self._categories()  # 初始化categories基本信息
        if self.mode=='train':
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            for i in tqdm([2,4,6,8,9]):
            # for i in tqdm(range(9)):
                scene_index=i
                # imgwidth,imgheight=self.get_imgshape(scene_index)
                annofile=os.path.join(self.root,PANDA_VIDEO_SEQS[scene_index],'gt/gt.txt')
                img_dir=os.path.join(self.root,PANDA_VIDEO_SEQS[scene_index],'img1/')
                print("processing ",PANDA_VIDEO_SEQS[scene_index])
                print(annofile)
                assert os.path.exists(annofile)
                gt_file = pd.read_csv(annofile, header=None)
                # gt_file = gt_file[gt_file[6] == 1]
                # gt_file = gt_file[gt_file[8] > 0.3]##min_visiblity=1/3
                gt_group = gt_file.groupby(0)
                gt_group_keys = list(gt_group.indices.keys())
                # length=len(gt_group_keys)
                # print("Train length:",length)
                # gt_group_keys = np.random.choice(gt_group_keys[:int(length*0.9)],int(length*0.5),replace=False)
                print("gt_group_keys",gt_group_keys)
                gt_group_keys = [key for key in gt_group_keys if key==100]
                for key in tqdm(gt_group_keys):
                    img_path=img_dir+'SEQ_'+'{0:02}'.format(scene_index)+"_"+str(key)+'.jpg'
                    print(img_path)
                    img=cv2.imread(img_path)
                    h,w=img.shape[0],img.shape[1]
                    os.system('cp '+img_path+' '+dataset_dir)
                    print("copy ",img_path," to ",dataset_dir," done!!!")
                    det = gt_group.get_group(key).values
                    det = np.array(det[:, 2:6])
                    for d in det:
                        annotation = self._annotation(d,1)
                        self.annotations.append(annotation)
                        self.ann_id+=1
                    self.images.append(self._image(img_path,h,w))
                    self.img_id+=1
        
        if self.mode=='valid':
            dataset_dir='/root/data/PANDA_detect/valid'
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            for i in tqdm([2]):
            # for i in tqdm([2,4,6,8,9]):
                scene_index=i+2
                # imgwidth,imgheight=self.get_imgshape(scene_index)
                annofile=self.root+PANDA_VIDEO_SEQS[scene_index]+'/gt/gt.txt'
                img_dir=self.root+PANDA_VIDEO_SEQS[scene_index]+'/img1/'
                print("processing ",PANDA_VIDEO_SEQS[scene_index])
                assert os.path.exists(annofile)
                gt_file = pd.read_csv(annofile, header=None)
                gt_file = gt_file[gt_file[6] == 1]
                gt_file = gt_file[gt_file[8] > 0.3]##min_visiblity=1/3
                gt_group = gt_file.groupby(0)
                gt_group_keys = list(gt_group.indices.keys())
                # length=len(gt_group_keys)
                # print("Val length:",length)
                # gt_group_keys = np.random.choice(gt_group_keys[int(length*0.9):],500,replace=False)
                # gt_group_keys = [key for key in gt_group_keys if (key-1)%100==0]

                for key in tqdm(gt_group_keys):
                    img_path=img_dir+'SEQ_'+'{0:02}'.format(scene_index)+'_'+str(key)+'.jpg'
                    os.system('cp '+img_path+' '+dataset_dir)
                    print("copy ",img_path," to ",dataset_dir," done!!!")
                    det = gt_group.get_group(key).values
                    det = np.array(det[:, 2:6])
                    for d in det:
                        annotation = self._annotation(d,1)
                        self.annotations.append(annotation)
                        self.ann_id+=1
                    
                    self.images.append(self._image(img_path,1280,1280))
                    self.img_id+=1

        instance = {}
        # instance['categories'] = self.categories
        instance['categories'] = [
        {"supercategory": "none", "id": 2, "name": 'full body'}]
        instance['info'] = 'mot2coco'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        return instance
  
    def save_coco_json(self, instance):
        os.makedirs('/root/data/gvision/dataset/mot/100/annotations',exist_ok=True)
        if self.mode=='train':
            save_path='/root/data/gvision/dataset/mot/100/annotations/instances_train.json'
        elif self.mode=='valid':
            save_path='/root/data/gvision/dataset/mot/PANDA_detect/annotations/instances_valid.json'
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))

panda2coco=PANDA2COCO(mode="train")###train or valid
instance=panda2coco.to_coco(dataset_dir='/root/data/gvision/dataset/mot/100/train')
panda2coco.save_coco_json(instance)
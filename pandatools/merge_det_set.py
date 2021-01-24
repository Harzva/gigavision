'''
written by ly 2020.7.20
set nms with Pixel extension.
first use score thresh to reduce some low score boxes, then set-nms.
'''
import os
import numpy as np
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
import math
import argparse

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

#0 score_thresh
#1 k
#2 b
#3 nms_threshold
#4 max_hw_ratio 长宽比
#5 min_hw_ratio
params=[
    [0.5,3,-2000,0.35,6,1], # 9
    [0.5, 1e-4,0,0.5,6,0.9], # 10
    [0.65, 4,-2600,0.6,6,0.9], # 11
    [0.65,5.5,750,0.6,6,1], # 12 ########set_nms=0.5
    [0.85,3,-2600,0.6,6,1/6], # 13
    [0.8,12,-2000,0.6,6,0.9], # 14
    [0.75,2.5,-3600,0.6,5,1] # 15
]

PANDA_TEST_SIZE=[
    [26583,14957], # 9
    [25306,14238], # 10
    [26583,14957],
    [32609,24457],
    [25654,14434],
    [25831,14533],
    [26583,14957]
]
UP_boundary=[
    0,0,3754,2004,4484,3733,3608
]

class MergeDet():
    def __init__(self,
                basepath='/root/data/PANDA_split_set/',
                scene_index=11,
                params=params,
                ext=2):
        self.resultpath = basepath + PANDA_VIDEO_SEQS[scene_index] + '.txt'
        self.savedir=basepath+'set_merged_dets'
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.scene_index=scene_index
        self.subimg_width = 1400
        self.subimg_height = 800
        self.imgWidth=PANDA_TEST_SIZE[scene_index-9][0]
        self.imgHeight=PANDA_TEST_SIZE[scene_index-9][1]
        self.params=params[scene_index-9]
        self.score_thresh=self.params[0]
        self.k=self.params[1]
        self.b=self.params[2]
        self.nms_threshold=self.params[3]
        self.max_hw_ratio=self.params[4]
        self.min_hw_ratio=self.params[5]
        self.ext = ext # Pixel extension

    def merge(self):
        print("Loading result file: ",self.resultpath)
        det_file=pd.read_csv(self.resultpath,header=None)
        det_group = det_file.groupby(0)
        det_group_keys = list(det_group.indices.keys())
        with open(self.savedir+'/'+PANDA_VIDEO_SEQS[self.scene_index] + '.txt','w') as f:
            for frameid in tqdm(det_group_keys):
                print("-"*60)
                print("processing frame ",frameid)
                dets=det_group.get_group(frameid)
                #frame_id,left,up,scale,x,y,w,h,score,number  columns=10
                det_left_up_group = dets.groupby([2,3])
                left_ups = det_left_up_group.indices.keys()
                big_frame_dets = np.array([[0.,0.,0.,0.,0.,0.]]) #[x,y,w,h,score,number]
                for left,up in left_ups:# get all dets in this small frame which left,up = left,up
                    dets = det_left_up_group.get_group((left,up)).values #frame_id,left,up,scale,x,y,w,h,score,number  columns=10
                    keeped_dets = self.keep_dets(np.array(dets))
                    if keeped_dets is None:
                        # print("now left up: ",left,up," no satisfied")
                        continue                    
                    #transfer to big frame coordinates, and concatenate to big_frame_dets
                    keeped_dets = self.get_ori_det(keeped_dets)
                    big_frame_dets = np.concatenate((big_frame_dets,keeped_dets),0)

                big_frame_dets = big_frame_dets[big_frame_dets[:,4]>self.score_thresh] #[x,y,w,h,score,number]
                big_frame_dets = big_frame_dets[self.set_cpu_nms(big_frame_dets,self.nms_threshold)] #ori:[x,y,w,h,score,number] 
                # big_frame_dets = self.remove_invalid_rects(big_frame_dets)
                for det in big_frame_dets:
                    x,y,w,h,score,_ = det
                    # print("detection box: ",x,y,w,h)
                    f.write(str(frameid)+','+'-1'+','+"{:.2f}".format(x)+','+"{:.2f}".format(y)+','
                                +"{:.2f}".format(w)+','+"{:.2f}".format(h)+','+"{:.2f}".format(score)+'\n')
        print("Merge Done!!!")

    def keep_dets(self,dets):
        '''
        :dets: frame_id,left,up,scale,x,y,w,h,score,number  columns=10
        :return dets in small frame [left,up,scale,x,y,w,h,score,number]   columns = 9
        '''
        keep_dets = []
        _,left,up,scale,_,_,_,_,_,_ = dets[0]
        print(left,up,scale)
        left,up=left*scale,up*scale
        print(left,up,scale)
        right = left + int(self.subimg_width/scale)
        down = up + int(self.subimg_height/scale)
        if up == UP_boundary[self.scene_index-9]:
            if left == 0:              # left_up_corner ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,number = det
                    if x+w >= self.subimg_width-self.ext or y+h >= self.subimg_height-self.ext:# if is out of right or down bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,number])
            elif right >= self.imgWidth-1:           # right_up_corner ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,number = det
                    if x <= 0+self.ext or y+h >= self.subimg_height-self.ext:# if is out of left or down bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,number])
            else:                      # up_bound ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,number = det
                    if x <= 0+self.ext or y+h >= self.subimg_height-self.ext or x+w >= self.subimg_width-self.ext:# if is out of left or down or right bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,number])
        elif left == 0:
            if down >= self.imgHeight-10:  # left_down_corner ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,number = det
                    if y <= 0+self.ext or x+w >= self.subimg_width-self.ext: # if is out of up or right bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,number])
            else:                      # left_bound ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,number = det
                    if y <= 0+self.ext or x+w >= self.subimg_width-self.ext or y+h >= self.subimg_height-self.ext:# if is out of up or right or down bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,number])
        elif down >= self.imgHeight-10: ####################################
            if right >= self.imgWidth-1: # right_down_corner ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,number = det
                    if x <= 0+self.ext or y <= 0+self.ext:
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,number])
            else:                      # down_bound ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,number = det
                    if x <= 0+self.ext or y <= 0+self.ext or x+w >= self.subimg_width-self.ext:
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,number])
        elif right >= self.imgWidth-1:   # right_broud ============
            for det in dets:
                _,left,up,scale,x,y,w,h,score,number = det
                if x <= 0+self.ext or y <= 0+self.ext or y+h >= self.subimg_height-self.ext:
                    continue
                else:
                    keep_dets.append([left,up,scale,x,y,w,h,score,number])
        else:                          # inner_part ============
            for det in dets:
                _,left,up,scale,x,y,w,h,score,number = det
                if x <= 0+self.ext or y <= 0+self.ext or x+w >= self.subimg_width-self.ext or y+h >= self.subimg_height-self.ext: ##################################
                    continue
                else:
                    keep_dets.append([left,up,scale,x,y,w,h,score,number])

        if len(keep_dets)>0:
            return np.array(keep_dets)
        else:
            return None
  
    def get_ori_det(self,dets):
        '''
        get coordinates in big frame
        dets = [left,up,scale,x,y,w,h,score,number]
        return dets in big frame det=[x,y,w,h,score,number]
        '''
        # print("dets:",dets[0])
        dets[:,3] = dets[:,0]+dets[:,3]/dets[:,2]
        dets[:,4] = dets[:,1]+dets[:,4]/dets[:,2]
        dets[:,5] = dets[:,5]/dets[:,2]
        dets[:,6] = dets[:,6]/dets[:,2]
        
        dets[dets<0] = 0

        coordinates = np.where((dets[:,3]+dets[:,5]) > self.imgWidth) #right>imgWidth
        dets[coordinates, 5] = self.imgWidth - dets[coordinates, 3]

        coordinates = np.where((dets[:,4]+dets[:,6]) > self.imgHeight)#down>imgHeight
        dets[coordinates, 6] = self.imgHeight - dets[coordinates, 4]
        return dets[:,3:]
 
    def set_cpu_nms(self, dets, thresh):
        """
        :dets: 二维numpy.ndarray, 每行6列 [x,y,w,h,score,number],需要保证在同一个set里的boxes的number是唯一的
        :return: bool 型numpy.ndarray, the index of keepded boxes.
        """
        def _overlap(det_boxes, basement, others):
            eps = 1e-8
            x1_basement, y1_basement, x2_basement, y2_basement \
                    = det_boxes[basement, 0], det_boxes[basement, 1], \
                    det_boxes[basement, 2], det_boxes[basement, 3]
            x1_others, y1_others, x2_others, y2_others \
                    = det_boxes[others, 0], det_boxes[others, 1], \
                    det_boxes[others, 2], det_boxes[others, 3]
            areas_basement = (x2_basement - x1_basement) * (y2_basement - y1_basement)
            areas_others = (x2_others - x1_others) * (y2_others - y1_others)
            xx1 = np.maximum(x1_basement, x1_others)
            yy1 = np.maximum(y1_basement, y1_others)
            xx2 = np.minimum(x2_basement, x2_others)
            yy2 = np.minimum(y2_basement, y2_others)
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas_basement + areas_others - inter + eps)
            return ovr
        scores = dets[:, 4]
        order = np.argsort(-scores)
        dets = dets[order] #按score从大到小排序
        #change to l t r d
        dets[:,2] = dets[:,2]+dets[:,0]
        dets[:,3] = dets[:,3]+dets[:,1]

        numbers = dets[:, -1]  #  set number
        keep = np.ones(len(dets)) == 1 # keep all at begining
        ruler = np.arange(len(dets)) # ruler = index of order # [0,1,2,3,4.....len]
        while ruler.size>0:
            basement = ruler[0]
            ruler=ruler[1:]
            num = numbers[basement]
            # calculate the body overlap
            overlap = _overlap(dets[:, :4], basement, ruler)
            indices = np.where(overlap > thresh)[0] 
            loc = np.where(numbers[ruler][indices] == num)[0] 
            # the mask won't change in the step
            mask = keep[ruler[indices][loc]]
            keep[ruler[indices]] = False
            keep[ruler[indices][loc][mask]] = True
            ruler[~keep[ruler]] = -1
            ruler = ruler[ruler>0]
        keep = keep[np.argsort(order)]
        return keep

    def remove_invalid_rects(self,dets):
        '''
        :input original set_nms crowd_det detections array
        :if the height_width ratio does not match ratio_thresh, remove\n
        :return boxes which to keep: [x,y,w,h,score,num] columns=6
        '''
        e=0.00001
        ## remove those does not match ratio_thresh or area_thresh
        keep = []
        for i in range(len(dets)):       
            left,up,w,h,score,_=dets[i]
            right = left+w
            down = up+h
            h_w_ratio=h/(w+e)
            if h_w_ratio>=self.min_hw_ratio and h_w_ratio<=self.max_hw_ratio:
                if  abs(up+self.b)/self.k>h:
                    # if not is_in([left,up,w,h],remove=REMOVE_AREA[self.scene_index-11]):
                    keep.append(dets[i])
        # print("len of keep :",len(keep))
        return np.array(keep)

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--scene",default=11,type=int,help="choose a scene to split!!!",required=True)
    args = parser.parse_args()
    mergedet = MergeDet(scene_index=args.scene)
    mergedet.merge()
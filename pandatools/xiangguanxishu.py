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

def keep_dets(self,dets):
        '''
        :dets: frame_id,left,up,scale,x,y,w,h,score,_  columns=10
        :return dets in small frame [left,up,scale,x,y,w,h,score,_]   columns = 9
        '''
        keep_dets = []
        _,left,up,scale,_,_,_,_,_,_ = dets[0]
        right = left + int(self.subimg_width/scale)
        down = up + int(self.subimg_height/scale)
        if up == UP_boundary[self.scene_index-9]:
            if left == 0:              # left_up_corner ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,_ = det
                    if x+w >= self.subimg_width-self.ext or y+h >= self.subimg_height-self.ext:# if is out of right or down bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,_])
            elif right >= self.imgWidth-1:           # right_up_corner ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,_ = det
                    if x <= 0+self.ext or y+h >= self.subimg_height-self.ext:# if is out of left or down bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,_])
            else:                      # up_bound ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,_ = det
                    if x <= 0+self.ext or y+h >= self.subimg_height-self.ext or x+w >= self.subimg_width-self.ext:# if is out of left or down or right bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,_])
        elif left == 0:
            if down >= self.imgHeight-10:  # left_down_corner ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,_ = det
                    if y <= 0+self.ext or x+w >= self.subimg_width-self.ext: # if is out of up or right bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,_])
            else:                      # left_bound ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,_ = det
                    if y <= 0+self.ext or x+w >= self.subimg_width-self.ext or y+h >= self.subimg_height-self.ext:# if is out of up or right or down bound
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,_])
        elif down >= self.imgHeight-10: ####################################
            if right >= self.imgWidth-1: # right_down_corner ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,_ = det
                    if x <= 0+self.ext or y <= 0+self.ext:
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,_])
            else:                      # down_bound ============
                for det in dets:
                    _,left,up,scale,x,y,w,h,score,_ = det
                    if x <= 0+self.ext or y <= 0+self.ext or x+w >= self.subimg_width-self.ext:
                        continue
                    else:
                        keep_dets.append([left,up,scale,x,y,w,h,score,_])
        elif right >= self.imgWidth-1:   # right_broud ============
            for det in dets:
                _,left,up,scale,x,y,w,h,score,_ = det
                if x <= 0+self.ext or y <= 0+self.ext or y+h >= self.subimg_height-self.ext:
                    continue
                else:
                    keep_dets.append([left,up,scale,x,y,w,h,score,_])
        else:                          # inner_part ============
            for det in dets:
                _,left,up,scale,x,y,w,h,score,_ = det
                if x <= 0+self.ext or y <= 0+self.ext or x+w >= self.subimg_width-self.ext or y+h >= self.subimg_height-self.ext: ##################################
                    continue
                else:
                    keep_dets.append([left,up,scale,x,y,w,h,score,_])

        if len(keep_dets)>0:
            return np.array(keep_dets)
        else:
            return None
if 391<=srcimageid<=420:#14otc
if 421<=srcimageid<=450:#15 nanshangongyuan
if 451<=srcimageid<=465:#16xiaoxue----------01
if 466<=srcimageid<=480:#16xiaoxue--------02
if 481<=srcimageid<=510:#17zhongguan
if 511<=srcimageid<=540:#18xilin-------01
if 541<=srcimageid<=555:#18xilin----------02
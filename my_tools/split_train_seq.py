import os
import cv2
import json
import copy
import glob
import numpy as np
from tqdm import tqdm

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
#     ""
#     '02_OCT_Habour',
#     '04_Primary_School',
#     '06_Xinzhongguan',
#     '08_Xili_Street_1',
#     '09_Xili_Street_2',
# ]
# train_seq_size=[
#     [26753,15052],
#     [26753,15052],
#     [26753,15052],
#     [34682,26012],
#     [31746,23810],
#     [26583,14957],
#     [25479,14335],
#     [26583,14957],
#     [26583,14957],
#     [25306,14238]
# ]
###split to three parts
'''
0 up_scale
1 mid_scale
2 bottom_scale
3 mid_up
4 bottom_mid
5 mid_up_gap
6 bottom_mid_gap
7 up_gap_h
8 up_gap_w
9 mid_gap_h
10 mid_gap_w
11 bottom_gap_w
12 up_boundary
'''
##split to two parts
'''
0 mid_scale
1 bottom_scale
2 bottom_mid
3 bottom_mid_gap
4 mid_gap_h
5 mid_gap_w 
6 bottom_gap_h
7 bottom_gap_w
8 up_boundary
'''#{'global_scale': -1, 'params': [0.6, 0.12, 6052, 1000, 4000]}
split_params=[ ##
    {'global_scale':-1,'params':[0.6,0.12,6052,1000,4000]},
    {'global_scale':0,'params':[1,0.8,0.2,5000,6800,600,1000,600,300,800,400,1500,3500]},#1280 1600 6400
    {'global_scale':0,'params':[1,0.8,0.2,5000,6800,600,1000,600,300,800,400,1500,4000]},
    {'global_scale':0.5,'params':[1200,600,0]},
    {'global_scale':0.4,'params':[1500,700,4000]},
    {'global_scale':-1,'params':[1,0.4,8200,800,800,400,1500,700,5000]},
    {'global_scale':-1,'params':[1,0.8,10000,600,600,300,800,400,3500]},
    {'global_scale':-1,'params':[1,0.3,6600,1000,800,400,2500,1300,4400]},
    {'global_scale':-1,'params':[1,0.3,5000,1000,800,400,2500,1300,3400]},
    {'global_scale':-1,'params':[1,0.4,8000,800,800,400,1500,700,5000]}
]

def get_frame_id(frame_path):
    return int(os.path.splitext(os.path.basename(frame_path))[0].split('_')[-1])

class Split_train_seq():
    def __init__(self,
                        basepath='/root/data/gvision/dataset/mot/train/',
                        outpath='/root/data/gvision/dataset/mot/PANDA_split_new/train',
                        scene_index=2,  ####int
                        subimg_size=1280,
                        thresh=0.1,
                        ):
        self.scene_index=scene_index
        self.basepath=basepath
        self.outpath=outpath
        self.subwidth = subimg_size
        self.subheight = subimg_size
        self.thresh = thresh
        self.imagepath=os.path.join(self.basepath,PANDA_VIDEO_SEQS[scene_index])##################
        self.annopath=os.path.join(self.basepath.replace('train','video_annos'),
                                PANDA_VIDEO_SEQS[scene_index],'tracks.json')
        self.outimagepath=os.path.join(self.outpath,PANDA_VIDEO_SEQS[scene_index],'img1')
        self.outannopath=os.path.join(self.outpath,PANDA_VIDEO_SEQS[scene_index],'gt')
        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outannopath):
            os.makedirs(self.outannopath)
        print("#"*60)
        print('Loading annotation json file: {}'.format(self.annopath))
        with open(self.annopath, 'r') as load_f:
            annodict = json.load(load_f)
        self.annos = annodict

    def split_scene(self):
        img_paths=glob.glob(self.imagepath+'/img1/*.jpg')
        img_paths= sorted(img_paths,key=get_frame_id)
        img_paths = [path for path in img_paths if get_frame_id(path)%40==0]
        current_frame=0
        for img in tqdm(img_paths):
            print("##### Start processing   ",img)
            current_frame=self.split_single(img,current_frame=current_frame)
        print("####Complete spliting scene: ",self.scene_index)

    def loadImg(self, imgpath):
        """
        :param imgpath: the path of image to load
        :return: loaded img object
        """
        if not os.path.exists(imgpath):
            print('Can not find {}, please check local dataset!'.format(imgpath))
            return None
        img = cv2.imread(imgpath)
        print("image",imgpath,"  has been load!!!")
        return img

    def split_single(self,imgpath,current_frame):
        img=self.loadImg(imgpath)
        imgHeight,imgWidth=img.shape[:2]
        current_frame_id=get_frame_id(imgpath)
        outbasename=self.outimagepath+'/SEQ_'+"{0:02d}".format(self.scene_index)+'_'
        ###################################################
        ### if  split to three parts
        # 0 up_scale
        # 1 mid_scale
        # 2 bottom_scale
        # 3 mid_up
        # 4 bottom_mid
        # 5 mid_up_gap
        # 6 bottom_mid_gap
        # 7 up_gap_h
        # 8 up_gap_w
        # 9 mid_gap_h
        # 10 mid_gap_w
        # 11 bottom_gap_w
        # 12 up_boundary {'global_scale': -1, 'params': [0.6, 0.12, 6052, 1000, 4000]}
        subimg_id=current_frame
        print(split_params[self.scene_index-1])
        if split_params[self.scene_index-1]['global_scale']==0:
            up_scale,mid_scale,bottom_scale,mid_up,bottom_mid,mid_up_gap,\
            bottom_mid_gap,up_gap_h,up_gap_w,mid_gap_h,mid_gap_w,\
            bottom_gap_w,up_boundary=split_params[self.scene_index-1]['params']
            ## first split the upper part
            up_bottom=mid_up+mid_up_gap
            left,up=0,up_boundary
            while left<imgWidth:
                if left+self.subwidth/up_scale>=imgWidth:
                    left=max(imgWidth-self.subwidth/up_scale,0)
                up=up_boundary
                while up<up_bottom:
                    if up+self.subheight/up_scale>=up_bottom:
                        up=max(up_bottom-self.subheight/up_scale,0)
                    subimg_id+=1
                    right=min(left+self.subwidth/up_scale,imgWidth)
                    down=min(up+self.subheight/up_scale,up_bottom)
                    coordinates=left,up,right,down
                    subimgname=outbasename+str(subimg_id)+'.jpg'
                    if self.annoSplit(current_frame_id,subimg_id,imgWidth,imgHeight,coordinates):
                        self.savesubimage(img,subimgname,coordinates,up_scale)
                    if up+self.subheight/up_scale>=up_bottom:
                        break
                    else:
                        up= up+self.subheight/up_scale-up_gap_h
                if left+self.subwidth/up_scale>=imgWidth:
                    break
                else:
                    left=left+self.subwidth/up_scale-up_gap_w
            ## second split the middle part
            left,up=0,mid_up
            mid_bottom=bottom_mid+bottom_mid_gap
            while left<imgWidth:
                if left+self.subwidth/mid_scale>=imgWidth:
                    left=max(imgWidth-self.subwidth/mid_scale,0)
                up=mid_up
                while up<mid_bottom:
                    if up+self.subheight/mid_scale>=mid_bottom:
                        up=max(mid_bottom-self.subheight/mid_scale,0)
                    subimg_id+=1
                    right=min(left+self.subwidth/mid_scale,imgWidth)
                    down=min(up+self.subheight/mid_scale,mid_bottom)
                    coordinates=left,up,right,down
                    subimgname=outbasename+str(subimg_id)+'.jpg'
                    if self.annoSplit(current_frame_id,subimg_id,imgWidth,imgHeight,coordinates):
                        self.savesubimage(img,subimgname,coordinates,mid_scale)
                    if up+self.subheight/mid_scale>=mid_bottom:
                        break
                    else:
                        up= up+self.subheight/mid_scale-mid_gap_h
                if left+self.subwidth/mid_scale>=imgWidth:
                    break
                else:
                    left=left+self.subwidth/mid_scale-mid_gap_w
            ## finally split the bottom part
            left,up=0,bottom_mid
            while left<imgWidth:
                if left+self.subwidth/bottom_scale>=imgWidth:
                    left=max(imgWidth-self.subwidth/bottom_scale,0)
                up=bottom_mid
                while up<imgHeight:
                    if up+self.subheight/bottom_scale>=imgHeight:
                        up=max(imgHeight-self.subheight/bottom_scale,0)
                    subimg_id+=1
                    right=min(left+self.subwidth/bottom_scale,imgWidth)
                    down=min(up+self.subheight/bottom_scale,imgHeight)
                    coordinates=left,up,right,down
                    subimgname=outbasename+str(subimg_id)+'.jpg'
                    if self.annoSplit(current_frame_id,subimg_id,imgWidth,imgHeight,coordinates):
                        self.savesubimage(img,subimgname,coordinates,bottom_scale)
                    if up+self.subheight/bottom_scale>=imgHeight:
                        break
                    else:
                        up= imgHeight-self.subheight/bottom_scale
                if left+self.subwidth/bottom_scale>=imgWidth:
                    break
                else:
                    left=left+self.subwidth/bottom_scale-bottom_gap_w
        #############################################################
        ## if  split to two parts
        # 0 mid_scale
        # 1 bottom_scale
        # 2 bottom_mid
        # 3 bottom_mid_gap
        # 4 mid_gap_h
        # 5 mid_gap_w 
        # 6 bottom_gap_h
        # 7 bottom_gap_w
        # 8 up_boundary
        elif split_params[self.scene_index-1]['global_scale']==-1:
            # print(len(split_params))
            # print(self.scene_index-1)
            mid_scale,bottom_scale,bottom_mid,bottom_mid_gap,mid_gap_h,\
            mid_gap_w,bottom_gap_h,bottom_gap_w,up_boundary=split_params[self.scene_index-1]['params']
            ## first split the middle part
            left,up=0,up_boundary
            mid_bottom=bottom_mid+bottom_mid_gap
            while left<imgWidth:
                if left+self.subwidth/mid_scale>=imgWidth:
                    left=max(imgWidth-self.subwidth/mid_scale,0)
                up=up_boundary
                while up<mid_bottom:
                    if up+self.subheight/mid_scale>=mid_bottom:
                        up=max(mid_bottom-self.subheight/mid_scale,0)
                    subimg_id+=1
                    right=min(left+self.subwidth/mid_scale,imgWidth)
                    down=min(up+self.subheight/mid_scale,mid_bottom)
                    coordinates=left,up,right,down
                    subimgname=outbasename+str(subimg_id)+'.jpg'
                    if self.annoSplit(current_frame_id,subimg_id,imgWidth,imgHeight,coordinates):
                        self.savesubimage(img,subimgname,coordinates,mid_scale)
                    if up+self.subheight/mid_scale>=mid_bottom:
                        break
                    else:
                        up= up+self.subheight/mid_scale-mid_gap_h
                if left+self.subwidth/mid_scale>=imgWidth:
                    break
                else:
                    left=left+self.subwidth/mid_scale-mid_gap_w
            ## sceond split the bottom part
            left,up=0,bottom_mid
            while left<imgWidth:
                if left+self.subwidth/bottom_scale>=imgWidth:
                    left=max(imgWidth-self.subwidth/bottom_scale,0)
                up=bottom_mid
                while up<imgHeight:
                    if up+self.subheight/bottom_scale>=imgHeight:
                        up=max(imgHeight-self.subheight/bottom_scale,0)
                    subimg_id+=1
                    right=min(left+self.subwidth/bottom_scale,imgWidth)
                    down=min(up+self.subheight/bottom_scale,imgHeight)
                    coordinates=left,up,right,down
                    subimgname=outbasename+str(subimg_id)+'.jpg'
                    if self.annoSplit(current_frame_id,subimg_id,imgWidth,imgHeight,coordinates):
                        self.savesubimage(img,subimgname,coordinates,bottom_scale)
                    if up+self.subheight/bottom_scale>=imgHeight:
                        break
                    else:
                        up= up+self.subheight/bottom_scale-bottom_gap_h
                if left+self.subwidth/bottom_scale>=imgWidth:
                    break
                else:
                    left=left+self.subwidth/bottom_scale-bottom_gap_w
        #############################################################
        ## directly split the original big frame
        else:
            global_scale=split_params[self.scene_index-1]['global_scale']
            gap_h,gap_w,up_boundary=split_params[self.scene_index-1]['params']
            left,up=0,up_boundary
            while left<imgWidth:
                if left+self.subwidth/global_scale>=imgWidth:
                    left=max(imgWidth-self.subwidth/global_scale,0)
                up=up_boundary
                while up<imgHeight:
                    if up+self.subheight/global_scale>=imgHeight:
                        up=max(imgHeight-self.subheight/global_scale,0)
                    subimg_id+=1
                    right=min(left+self.subwidth/global_scale,imgWidth)
                    down=min(up+self.subheight/global_scale,imgHeight)
                    coordinates=left,up,right,down
                    subimgname=outbasename+str(subimg_id)+'.jpg'
                    if self.annoSplit(current_frame_id,subimg_id,imgWidth,imgHeight,coordinates):
                        self.savesubimage(img,subimgname,coordinates,global_scale)
                    if up+self.subheight/global_scale>=imgHeight:
                        break
                    else:
                        up= up+self.subheight/global_scale-gap_h
                if left+self.subwidth/global_scale>=imgWidth:
                    break
                else:
                    left=left+self.subwidth/global_scale-gap_w
        print("Complete spliting Frame ",current_frame_id,"!!!")
        return subimg_id

    def annoSplit(self,current_frame,sub_frame_id,imgwidth, imgheight,coordinates):
        split=False
        with open(self.outannopath+'/gt.txt','a') as f:
            for track_dict in self.annos:
                track_id = track_dict["track id"]
                for frame_dict in track_dict["frames"]:
                    frame_id = frame_dict["frame id"]
                    if frame_id  == current_frame:
                    # process  this picture and anno
                        rect = frame_dict["rect"]
                        occ = frame_dict["occlusion"]
                        # print(rect)
                        if self.judgeRect(rect,imgwidth, imgheight,coordinates):
                            
                            rect=self.restrainRect(rect, imgwidth, imgheight,coordinates)#change rect size in subimg
                            x, y, w, h = self.RectDict2List(rect, self.subwidth, self.subheight, scale=1, mode='tlwh')#to mot
                            if occ == 'normal' or occ=='hide':  # or occ=='serious hide':
                                split=True
                                f.writelines([str(sub_frame_id), ',', str(track_id), ',', "{:.2f}".format(x), ',', "{:.2f}".format(y), ',', "{:.2f}".format(w), ',',
                                            "{:.2f}".format(h), ',', '1', ',', '1', ',', '1', '\n'])
        return split
    
    def judgeRect(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        # xmin = int(rectdict['tl']['x'] * imgwidth)
        xmin = rectdict['tl']['x'] * imgwidth
        # ymin = int(rectdict['tl']['y'] * imgheight)
        ymin = rectdict['tl']['y'] * imgheight
        # xmax = int(rectdict['br']['x'] * imgwidth)
        xmax = rectdict['br']['x'] * imgwidth
        # ymax = int(rectdict['br']['y'] * imgheight)
        ymax = rectdict['br']['y'] * imgheight
        square = (xmax - xmin) * (ymax - ymin)

        if (xmax <= left or right <= xmin) and (ymax <= up or down <= ymin):
            intersection = 0
        else:
            lens = min(xmax, right) - max(xmin, left)
            wide = min(ymax, down) - max(ymin, up)
            intersection = lens * wide

        return intersection and intersection / (square + 1e-5) > self.thresh

    def restrainRect(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        # xmin = int(rectdict['tl']['x'] * imgwidth)
        xmin = rectdict['tl']['x'] * imgwidth
        # ymin = int(rectdict['tl']['y'] * imgheight)
        ymin = rectdict['tl']['y'] * imgheight
        # xmax = int(rectdict['br']['x'] * imgwidth)
        xmax = rectdict['br']['x'] * imgwidth
        # ymax = int(rectdict['br']['y'] * imgheight)
        ymax = rectdict['br']['y'] * imgheight
        xmin = max(xmin, left)
        xmax = min(xmax, right)
        ymin = max(ymin, up)
        ymax = min(ymax, down)
        return {
            'tl': {
                'x': (xmin - left) / (right - left),
                'y': (ymin - up) / (down - up)
            },
            'br': {
                'x': (xmax - left) / (right - left),
                'y': (ymax - up) / (down - up)
            }
        }

    def restrain_between_0_1(self,values_list):
        return_list = []
        for value in values_list:
            if value < 0:
                new_value = 0.0
            elif value > 1:
                new_value = 1.0
            else:
                new_value = value
            return_list.append(new_value)
        return return_list

    def RectDict2List(self,rectdict, imgwidth, imgheight, scale, mode='tlbr'):
        x1, y1, x2, y2 = self.restrain_between_0_1([rectdict['tl']['x'], rectdict['tl']['y'],
                                            rectdict['br']['x'], rectdict['br']['y']])
        # xmin = int(x1 * imgwidth * scale)
        xmin = x1 * imgwidth * scale
        # ymin = int(y1 * imgheight * scale)
        ymin = y1 * imgheight * scale
        # xmax = int(x2 * imgwidth * scale)
        xmax = x2 * imgwidth * scale
        # ymax = int(y2 * imgheight * scale)
        ymax = y2 * imgheight * scale

        if mode == 'tlbr': #bottom right
            return xmin, ymin, xmax, ymax
        elif mode == 'tlwh': #weight height
            return xmin, ymin, xmax - xmin, ymax - ymin

    def savesubimage(self, img, subimgname, coordinates,scale):
        left, up, right, down = int(coordinates[0]),int(coordinates[1]),int(coordinates[2]),int(coordinates[3])
        subimg = copy.deepcopy(img[up: down, left: right])
        # print(subimg.shape)
        # print(scale)
        subimg=cv2.resize(subimg,None,fx=scale,fy=scale)
        # print(subimg.shape)
        print("Saving img ",subimgname)
        cv2.imwrite(subimgname, subimg)

if __name__=="__main__":
    # video_split=Split_train_seq(scene_index=7)
    # video_split.split_scene()
    for i in [8,9]:
        video_split=Split_train_seq(scene_index=i)
        video_split.split_scene()
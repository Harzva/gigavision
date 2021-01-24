from genericpath import exists
import json
import cv2
import os
import numpy as np
import random
from operator import itemgetter
import bisect 


"coco_dataset_visual(d2)，gt_visual(show_annos)，result_visual"

basename="/root/data/rubzz/ruby/xml/coco/images"
def coco_dataset_visual(basename="/root/data/rubzz/ruby/xml/coco",
                # annos_path="/root/data/rubzz/ruby/PANDA-Toolkit-master/ruby/my_eva_coco_all.json",
                annos_path="/root/data/rubzz/ruby/xml/coco/all/eva_coco_person.json",
                # annos_path="/root/data/rubzz/ruby/xml/coco/all/eva_coco_all.json",
                # annos_path="/root/data/rubzz/ruby/xml/coco/my_eva_coco_test.json",
                savepath=os.path.join(basename,"annos_in_image")):
# def coco_dataset_visual(basename="/root/data/rubzz/ruby/ruby_output4/the_6_we_choose_person/img",
#                 annos_path="/root/data/rubzz/ruby/ruby_output4/the_6_we_choose_person/the_6_we_choose_person_anno.json",
#                 savepath="/root/data/rubzz/ruby/ruby_output4/annos_in_image"):
    with open(annos_path,'r') as load_f:
        coco_dicts= json.load(load_f)
    random.seed(1)

    # for j,images_dict in  enumerate(random.sample(coco_dicts["images"],10)):
    for j,images_dict in  enumerate(coco_dicts["images"]):
        file_name=images_dict["file_name"]
        # file_name="images/IMG_14_02.jpg"
        if j==2:
            break
        print("{}\t{}-------------------{}".format(file_name,j,10),flush=True)
        image_id=images_dict["id"]
        print(file_name)
        print(os.path.join(basename,file_name))
        img=cv2.imread(os.path.join(basename,file_name))
        # print(img)
        cate=[]
        id_1,id_2,id_3,id_4=0,0,0,0
        for annos_dict in coco_dicts["annotations"]:
            i=0
            if annos_dict["image_id"]==image_id:
                i+=1
                cate.append(annos_dict["category_id"])
                xmin, ymin, w , h = annos_dict["bbox"]
                xmax,ymax=xmin+w,ymin+h
                xmin, ymin,xmax,ymax=int(xmin),int(ymin),int(xmax),int(ymax)
                if annos_dict["category_id"]==1:#green
                    id_1+=1
                    img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,255,0), 2,lineType=2)
                    # cv2.putText(img, '{}'.format(annos_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 5, (138,255,0), 18)
                if annos_dict["category_id"]==2:#pink
                    id_2+=1
                    img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,0,255), 2,lineType=2)
                    # cv2.putText(img, '{}'.format(annos_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 5, (138,0,255), 18)
                if annos_dict["category_id"]==3:#purple
                    id_3+=1
                    img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,46,46), 2,lineType=2)
                    # cv2.putText(img, '{}'.format(annos_dict["category_id"]), (xmin+200,ymin), cv2.FONT_HERSHEY_COMPLEX, 5, (255,46,46), 18)
                if annos_dict["category_id"]==4:#grey
                    id_4+=1
                    img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (131,131,131), 2,lineType=2)
                    # cv2.putText(img, '{}'.format(annos_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (131,131,131), 18)
        cv2.putText(img, f"c1:{id_1}", (2500,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,255,0), 35)
        cv2.putText(img, f"c2:{id_2}", (5000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,0,255), 35)
        cv2.putText(img, f"c3:{id_3}", (8000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (255,46,46), 35)
        cv2.putText(img, f"c4:{id_4}", (11000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (131,131,131), 35)
        os.makedirs(os.path.join(savepath,"images"),exist_ok=True)
        a=os.path.join(savepath,"{}".format(file_name))
        print(a)
        imgheight, imgwidth = img.shape[:2]
        scale =0.05
        img = cv2.resize(img, (int(imgwidth * scale), int(imgheight * scale)))

        cv2.imwrite(a,img)
        # /root/data/rubzz/ruby/ruby_output2/train_person_fafaxue/img/01_University_Canteen_IMG_01_19___0.2546__15404__4504.jpg
# coco_dataset_visual()      
        
# def gt_visual(annos_path):
""" {
    "image_id": 391,
    "category_id": 3,
    "bbox": [
      24704,
      7360,
      2046,
      1
    ],
    "score": 1.0
  }
    "14_OCT_Habour_IMG_14_01___0.5__0__0.jpg": {
    "image size": {
      "height": 2049,
      "width": 1025
    },
    "image id": 1,"""
def split_result_visual(image_folder_test,result_path,annos_path,savepath,imgfilters,mode,num=10):
    with open(annos_path,'r') as load_f:
        annos_dict= json.load(load_f)
    with open(result_path,'r') as load_r:
        result_list= json.load(load_r)
    random.seed(1)
    tempannos={}
    if imgfilters:
        for imgfilter in imgfilters:
            tempannos.update({i:j for i,j in annos_dict.items() if imgfilter in i })
        annos_dict=tempannos
    for j,(file_name,images_dict) in  enumerate(random.sample(annos_dict.items(),num)):
        img=cv2.imread(os.path.join(image_folder_test,file_name))
        rectange_list=[x for x in result_list if x["score"]>0.3]
        # len_per_image=len(rectange_list)
        # # print(img)
        # # print(type(img))
        # draw_rectange(img,savepath,file_name,rectange_list,len_per_image,thickness=20,output_prefix="testone3",mode=mode)
        len_per_image=0
        id_1,id_2,id_3,id_4=0,0,0,0
        for l,result_dict in  enumerate(result_list):
            if result_dict["image_id"]==images_dict["image id"]:
                len_per_image+=1
                if mode=="xyxy":
                    xmin, ymin,xmax,ymax=int(result_dict["bbox"][0]),int(result_dict["bbox"][1]),int(result_dict["bbox"][2]),int(result_dict["bbox"][3])
                if mode=="xywh":
                    xmin, ymin,w,h=int(result_dict["bbox"][0]),int(result_dict["bbox"][1]),int(result_dict["bbox"][2]),int(result_dict["bbox"][3])
                    xmax,ymax=xmin+w,ymin+h
                if result_dict["category_id"]==1:#green
                    id_1+=1
                    # print(img.shape, (xmin, ymin), (xmax, ymax))
                    img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,255,0), 2,lineType=2)
                    cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (138,255,0), 2)
                    cv2.putText(img, '{:.2f}'.format(result_dict["score"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (138,255,0), 2)
                if result_dict["category_id"]==2:#pink
                    id_2+=1
                    img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,0,255), 2,lineType=2)
                    cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (138,0,255), 2)
                    cv2.putText(img, '{:.2f}'.format(result_dict["score"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (138,0,255), 2)
                if result_dict["category_id"]==3:#blue
                    id_3+=1
                    img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,46,46), 2,lineType=2)
                    cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255,46,46), 2)
                if result_dict["category_id"]==4:#grey
                    id_4+=1
                    img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (131,131,131),2,lineType=2)
                    cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (131,131,131), 2)
        cv2.putText(img, f"c1:{id_1}", (250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (138,0,255),2)
        cv2.putText(img, f"c2:{id_2}", (400,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (138,255,0),2)  
        cv2.putText(img, f"c3:{id_3}", (800,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,46,46),2)      
        os.makedirs(savepath,exist_ok=True)
        a=os.path.join(savepath,"test_person_one2_{}".format(file_name.replace("/","_")))
        print(a)
        cv2.imwrite(a,img)
def draw_rectange(img,savepath,file_name,rectange_list,len_per_image,thickness,output_prefix,mode="xywh",):
    cate=[]
    id_1,id_2,id_3,id_4=0,0,0,0
    zeros_ = np.zeros((img.shape), dtype=np.uint8)
    for j,result_dict in enumerate(rectange_list):
        # if j%50==0:
        #     # print(f"{file_name}------------{j}")
        if mode=="xyxy":
            xmin, ymin,xmax,ymax=int(result_dict["bbox"][0]),int(result_dict["bbox"][1]),int(result_dict["bbox"][2]),int(result_dict["bbox"][3])
        if mode=="xywh":
            xmin, ymin,w,h=int(result_dict["bbox"][0]),int(result_dict["bbox"][1]),int(result_dict["bbox"][2]),int(result_dict["bbox"][3])
            xmax,ymax=xmin+w,ymin+h
        if result_dict["category_id"]==1:#green
            id_1+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(138,255,0), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,255,0),thickness,lineType=18)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 14, (138,255,0), 25)
                cv2.putText(img, '{:.2f}'.format(result_dict["score"]),(xmin,int((ymin+ymax)/2)), cv2.FONT_HERSHEY_COMPLEX, 4, (138,255,0), 4)
                # cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(xmin,int((ymin+ymax)/2)), cv2.FONT_HERSHEY_COMPLEX, 4, (138,0,255), 20)

        if result_dict["category_id"]==2:#pink
            id_2+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(138,0,255), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,0,255), thickness,lineType=18)
                cv2.putText(img, '{:.2f}'.format(result_dict["score"]),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,0,255), 4)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX,14, (138,0,255), 25)
        if result_dict["category_id"]==3:#purple
            id_3+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(255,46,46), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,46,46), thickness,lineType=18)
                cv2.putText(img, '{:.2f}'.format(result_dict["score"]),(int((xmin+xmax)/2),ymin), cv2.FONT_HERSHEY_COMPLEX,4, (255,46,46), 4)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 14, (255,46,46), 25)
        if result_dict["category_id"]==4:#grey
            id_4+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(240,215,39), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (240,215,39), thickness,lineType=18)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX,8, (138,0,255), 20)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,int((ymin+ymax)/2)), cv2.FONT_HERSHEY_COMPLEX, 8, (138,0,255), 20)
        if thickness==-1:
            # alpha 为第一张图片的透明度
            alpha = 1
            # beta 为第二张图片的透明度
            beta = 0.01
            gamma = 0
            # cv2.addWeighted 将原始图片与 mask 融合
            img= cv2.addWeighted(img, alpha, np.array(zeros_mask), beta, gamma)
    cv2.putText(img, "len:{}".format(len_per_image,), (100,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,0,255), 35)
    cv2.putText(img, f"c1:{id_1}", (2500,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,255,0), 35)
    cv2.putText(img, f"c2:{id_2}", (5000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,0,255), 35)
    cv2.putText(img, f"c3:{id_3}", (8000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (255,46,46), 35)
    cv2.putText(img, f"c4:{id_4}", (11000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (131,131,131), 35)
    os.makedirs(savepath,exist_ok=True)
    img=cv2.resize(img,(int(img.shape[1]*0.1),int(img.shape[0]*0.1)))
    print(os.path.join(savepath,"{}{}_len_{}.jpg".format(file_name.replace("/","_")[:-4],output_prefix,len_per_image)))
    cv2.imwrite(os.path.join(savepath,"{}{}_len_{}.jpg".format(file_name.replace("/","_")[:-4],output_prefix,len_per_image)),img,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])

def draw_rectange1(img,savepath,file_name,rectange_list,len_per_image,thickness,output_prefix,mode="xywh"):
    cate=[]
    id_1,id_2,id_3,id_4=0,0,0,0
    zeros_ = np.zeros((img.shape), dtype=np.uint8)
    for j,result_dict in enumerate(rectange_list):
        # if j%50==0:
        #     # print(f"{file_name}------------{j}")
        if mode=="xyxy":
            xmin, ymin,xmax,ymax=int(result_dict["bbox"][0]),int(result_dict["bbox"][1]),int(result_dict["bbox"][2]),int(result_dict["bbox"][3])
        if mode=="xywh":
            xmin, ymin,w,h=int(result_dict["bbox"][0]),int(result_dict["bbox"][1]),int(result_dict["bbox"][2]),int(result_dict["bbox"][3])
            xmax,ymax=xmin+w,ymin+h
        if result_dict["category_id"]==1:#green
            id_1+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(138,255,0), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,255,0),thickness,lineType=1)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,255,0), 15)
                # cv2.putText(img, '{}'.format(result_dict["score"],'3.f'),(xmin,int((ymin+ymax)/2)), cv2.FONT_HERSHEY_COMPLEX, 2, (138,255,0), 20)
                # cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(xmin,int((ymin+ymax)/2)), cv2.FONT_HERSHEY_COMPLEX, 4, (138,0,255), 20)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,255,0), 15)

        if result_dict["category_id"]==2:#pink
            id_2+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(138,0,255), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,0,255), thickness,lineType=1)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,255,0), 15)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(int((xmin+xmax)*2/4),ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,0,255), 2)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX,14, (138,0,255), 25)
        if result_dict["category_id"]==3:#purple
            id_3+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(255,46,46), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,46,46), thickness,lineType=1)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(int((xmin+xmax)/2),ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (255,46,46), 2)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 14, (255,46,46), 25)
        if result_dict["category_id"]==4:#grey
            id_4+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(240,215,39), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (240,215,39), thickness,lineType=12)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX,8, (138,0,255), 2)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,int((ymin+ymax)/2)), cv2.FONT_HERSHEY_COMPLEX, 8, (138,0,255), 20)
        if thickness==-1:
            # alpha 为第一张图片的透明度
            alpha = 1
            # beta 为第二张图片的透明度
            beta = 0.01
            gamma = 0
            # cv2.addWeighted 将原始图片与 mask 融合
            img= cv2.addWeighted(img, alpha, np.array(zeros_mask), beta, gamma)
    cv2.putText(img, "len:{}".format(len_per_image,), (100,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,0,255), 35)
    cv2.putText(img, f"c1:{id_1}", (2500,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,255,0), 35)
    cv2.putText(img, f"c2:{id_2}", (5000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,0,255), 35)
    cv2.putText(img, f"c3:{id_3}", (8000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (255,46,46), 35)
    cv2.putText(img, f"c4:{id_4}", (11000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (131,131,131), 35)
    os.makedirs(savepath,exist_ok=True)
    # img=cv2.resize(img,(int(img.shape[1]*0.1),int(img.shape[0]*0.1)))
    savepath="/root/data/testvisual"
    print(f"save {file_name} as:",os.path.join(savepath,"len_{}{}_{}".format(len_per_image,output_prefix,file_name.replace("/","_"))))
    cv2.imwrite(os.path.join(savepath,"len_{}{}_{}".format(len_per_image,output_prefix,file_name.replace("/","_"))),img,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])

def merge_result_visual(image_folder_test,result_path,annos_path,savepath,output_prefix,imgfilters,test=None,mode="xywh",num=10,score=0.0):
    with open(annos_path,'r') as load_f:
        annos_dict= json.load(load_f)
    with open(result_path,'r') as load_r:
        result_list= json.load(load_r)
    print(len(result_list[1]))
    os.makedirs(savepath,exist_ok=True)
    random.seed(0)
    # imgfilters=["15_24"]
    if test:
        print("test------start")
        tempannos={}
        if imgfilters:
            for imgfilter in imgfilters:
                tempannos.update({i:j for i,j in annos_dict.items() if imgfilter in i })
            annos_dict=tempannos
        # print(annos_dict)
        # print(result_list[1])
    for j,(file_name,images_dict) in enumerate(annos_dict.items()):
        id= file_name.split('.')[0][-5:]
        if test:
            img=cv2.imread(os.path.join(image_folder_test,file_name))
            rectange_list=[x for x in result_list if x["image_id"]==images_dict["image id"] and x["score"]>score]

            rectange_list1=[x for x in rectange_list if x["category_id"]==1]
            rectange_list2=[x for x in rectange_list if x["category_id"]==2]
            rectange_list3=[x for x in rectange_list if x["category_id"]==3]
            rectange_list4=[x for x in rectange_list if x["category_id"]==4]
            
            # rectange_list1= sorted(rectange_list1, key=itemgetter('score'),reverse=True)[0:500]
            # rectange_list2= sorted(rectange_list2, key=itemgetter('score'),reverse=True)[0:500]
            # rectange_list3= sorted(rectange_list3, key=itemgetter('score'),reverse=True)[0:500]
            # rectange_list4= sorted(rectange_list4, key=itemgetter('score'),reverse=True)[0:500]

            rectange_list=rectange_list1+rectange_list2+rectange_list3+rectange_list4
            len_per_image=len(rectange_list)
            draw_rectange(img,savepath,file_name,rectange_list,len_per_image,thickness=10,output_prefix=output_prefix,mode=mode)
        elif not test and (id in imgfilters):
            img=cv2.imread(os.path.join(image_folder_test,file_name))
            rectange_list=[x for x in result_list if x["image_id"]==images_dict["image id"] and x["score"]>score]
            len_per_image=len(rectange_list)
            draw_rectange(img,savepath,file_name,rectange_list,len_per_image,thickness=2,output_prefix=output_prefix,mode=mode)
def main():
    # coco_dataset_visual()
    """merge----inference"""
    test=1
    imgfilters=["15_27","14_02","16_01","17_23","18_01","14_30","14_17","17_21"]
    image_prefix=""
    merge_result_visual(image_folder_test="/root/data/gvision/dataset/raw_data/image_test",
            # result_path="/root/data/gvision/final_merge/final5/submit_dyy_all.json",
            # result_path="/root/data/gvision/post/hzh/nm0.99send.json",
            result_path="/root/data/rubzz/fengwei/fafaresult/json/sub1.json",
            # result_path="/root/data/gvision/post/hzh/nms0.99_nofs_head.json",
            # result_path="/root/data/gvision/post/hzh/dyytour.json",
            # result_path="/root/data/gvision/post/hzh/nms_fs_head.json",
            # result_path="/root/data/gvision/final_merge/final5/submit_xml.json",
            # result_path="/root/data/gvision/final_merge/final4/submit_123_4_all.json",
            # result_path="/root/data/EfficientDet/output/fullbody_fafaxue_all.json",
            annos_path="/root/data/gvision/dataset/raw_data/image_annos/person_bbox_test.json",
            savepath="/root/data/gvision/post/visual",
            output_prefix=image_prefix,
            imgfilters=imgfilters,
            test=test,
            mode="xywh",
            num=10)#mode: input_bbox mode
    # basename="/root/data/gvision/dataset/raw_data"
    # merge_result_visual(image_folder_test="/root/data/gvision/dataset/raw_data/image_test",
    #     result_path="/root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/coco_results/full_emnms_crowdet.json",
    #                 annos_path="/root/data/gvision/dataset/raw_data/image_annos/person_bbox_test.json",
    #                 savepath="/root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/visual/my_merge/fullbody",
    #                 output_prefix="test_all_emsnmssc0.75",
    #                 mode="xywh",
    #                 num=1)#mode: input_bbox mode

    # merge_result_visual(image_folder_test="/root/data/gvision/dataset/raw_data/image_test",
    #                 result_path="/root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/coco_results/merge_testone3.json",
    #                 annos_path="/root/data/gvision/dataset/raw_data/image_annos/person_bbox_test.json",
    #                 savepath="/root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/visual/my_merge",
    #                 output_prefix="testone11",
    #                 mode="xywh",
    #                 num=1)#mode: input_bbox mode
    #     test_json= "/root/data/rubzz/ruby/ruby_output/test/person/split_test_method2_person.json"
    # test_image_path="/root/data/rubzz/ruby/ruby_output/test/person/split_test_method2_person"

    # """split----predict"""
    # split_result_visual(
    #             image_folder_test="/root/data/rubzz/ruby/ruby_output3/split_test_person_fafaxue/img",
    #             result_path="/root/data/gvision/mmdetection-master/workdir/detectors_1_2_fafaxue/output/detectors_ep30_12_fafaxue_bbox.json",
    #             annos_path="/root/data/rubzz/ruby/ruby_output3/split_test_person_fafaxue/test_person_fafaxue.json",
    #             savepath="/root/data/gvision/mmdetection-master/workdir/detectors_1_2_fafaxue/visual",
    #             imgfilters=["15_01"],
    #             mode="xywh",num=10)

    # split_result_visual(
    #     image_folder_test="/root/data/rubzz/ruby/ruby_output3/split_test_person_fafaxue/img",
    #     result_path="/root/data/gvision/mmdetection-master/workdir/detectors_123_fafaxue_resume/output/detectors_123resume_fafaxue_bbox.json",
    #     annos_path="/root/data/rubzz/ruby/ruby_output3/split_test_person_fafaxue/test_person_fafaxue.json",
    #     savepath="/root/data/gvision/mmdetection-master/workdir/detectors_123_fafaxue_resume/visual",
    #     imgfilters=["15_01"],
    #     mode="xywh",num=10)          

    # """merge----predict"""
    # basename="/root/data/gvision/dataset/raw_data"
    # result_visual(result_path="/root/data/gvision/dataset/d2_output/my_pv_mask/my_predict/pre_merge_result.json",
    #                 annos_path=os.path.join(basename,"image_annos","test_14_15_17.json"),
    #                 savepath="/root/data/gvision/dataset/d2_output/my_pv_mask/my_predict_merge_visual",
    #                 mode="xywh",num=3)
if __name__ == "__main__":
    main()

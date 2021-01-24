from genericpath import exists
import json
import cv2
import os
import numpy as np
import random

from operator import itemgetter

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
        if result_dict["category_id"]==1:#yellow
            id_1+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(0,255,255), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,255), thickness,lineType=1)
                cv2.putText(img, '{:.2f}'.format(result_dict["score"]),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255), 1)
        if result_dict["category_id"]==2:#green
            id_2+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(0,255,0), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0),thickness,lineType=1)
                cv2.putText(img, '{:.2f}'.format(result_dict["score"]),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

        if result_dict["category_id"]==3:#red
            id_3+=1 
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(0,0,255), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), thickness,lineType=1)
                cv2.putText(img, '{:.2f}'.format(result_dict["score"]),(int((xmin+xmax)/2),ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        if result_dict["category_id"]==4:#blue
            id_4+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(255,0,0), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), thickness,lineType=1)
                cv2.putText(img, '{:.2f}'.format(result_dict["score"]),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX,1, (255,0,0), 1)
        if thickness==-1:
            # alpha 为第一张图片的透明度
            alpha = 1
            # beta 为第二张图片的透明度
            beta = 0.01
            gamma = 0
            # cv2.addWeighted 将原始图片与 mask 融合
            img= cv2.addWeighted(img, alpha, np.array(zeros_mask), beta, gamma)
    # print(file_name,len_per_image,id_1,id_2,id_3,id_4)
    cv2.putText(img, "len:{}".format(len_per_image,), (100,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (138,0,255), 35)
    cv2.putText(img, f"c1:{id_1}", (2500,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (0,255,255), 35)
    cv2.putText(img, f"c2:{id_2}", (5000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (0,255,0), 35)
    cv2.putText(img, f"c3:{id_3}", (8000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (0,0,255), 35)
    cv2.putText(img, f"c4:{id_4}", (11000,500), cv2.FONT_HERSHEY_SIMPLEX, 18, (255,0,0), 35)
    os.makedirs(savepath,exist_ok=True)
    # img=cv2.resize(img,(int(img.shape[1]*0.1),int(img.shape[0]*0.1)))
    print(f"save {file_name} as:",os.path.join(savepath,"len_{}{}_{}".format(len_per_image,output_prefix,file_name.replace("/","_"))))
    cv2.imwrite(os.path.join(savepath,"len_{}{}_{}".format(len_per_image,output_prefix,file_name.replace("/","_"))),img)
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
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,255,0),thickness,lineType=15)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,255,0), 15)
                # cv2.putText(img, '{}'.format(result_dict["score"],'3.f'),(xmin,int((ymin+ymax)/2)), cv2.FONT_HERSHEY_COMPLEX, 2, (138,255,0), 20)
                # cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(xmin,int((ymin+ymax)/2)), cv2.FONT_HERSHEY_COMPLEX, 4, (138,0,255), 20)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,255,0), 10)

        if result_dict["category_id"]==2:#pink
            id_2+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(138,0,255), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,0,255), thickness,lineType=15)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,255,0), 15)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(int((xmin+xmax)*2/4),ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (138,0,255), 10)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX,14, (138,0,255), 25)
        if result_dict["category_id"]==3:#purple
            id_3+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(255,46,46), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,46,46), thickness,lineType=15)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(int((xmin+xmax)/2),ymin), cv2.FONT_HERSHEY_COMPLEX, 4, (255,46,46), 10)
                # cv2.putText(img, '{}'.format(result_dict["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 14, (255,46,46), 25)
        if result_dict["category_id"]==4:#grey
            id_4+=1
            if thickness==-1:
                zeros_mask=cv2.rectangle(zeros_, (xmin, ymin), (xmax, ymax),(240,215,39), thickness)
            else:
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (240,215,39), thickness,lineType=12)
                cv2.putText(img, '{}'.format(round(result_dict["score"],2)),(xmin,ymin), cv2.FONT_HERSHEY_COMPLEX,8,(240,215,39), 10)
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
    savepath="/root/data/testvisual"
    print(f"save {file_name} as:",os.path.join(savepath,"len_{}{}_{}".format(len_per_image,output_prefix,file_name.replace("/","_"))))
    cv2.imwrite(os.path.join(savepath,"len_{}{}_{}".format(len_per_image,output_prefix,file_name.replace("/","_"))),img,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])

def merge_result_visual(image_folder_test,result_path,annos_path,savepath,output_prefix,imgfilters,test=None,mode="xywh",num=10):
    with open(annos_path,'r') as load_f:
        annos_dict= json.load(load_f)
    with open(result_path,'r') as load_r:
        result_list= json.load(load_r)
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
            # rectange_list=[x for x in result_list if x["image_id"]==images_dict["image id"] and x["score"]>0.1]
            rectange_list=[x for x in result_list if x["image_id"]==images_dict["image id"]]

            rectange_list1=[x for x in result_list if x["category_id"]==1]
            rectange_list2=[x for x in result_list if x["category_id"]==2]
            rectange_list3=[x for x in result_list if x["category_id"]==3]
            rectange_list4=[x for x in result_list if x["category_id"]==4]
            
            rectange_list1= sorted(rectange_list1, key=itemgetter('score'),reverse=True)[0:500]
            print(len(rectange_list1))
            rectange_list2= sorted(rectange_list2, key=itemgetter('score'),reverse=True)[0:500]
            print(len(rectange_list2))
            rectange_list3= sorted(rectange_list3, key=itemgetter('score'),reverse=True)[0:500]
            print(len(rectange_list3))
            rectange_list4= sorted(rectange_list4, key=itemgetter('score'),reverse=True)[0:500]
            print(len(rectange_list4))
            rectange_list=rectange_list3#rectange_list1#+rectange_list2+rectange_list3+rectange_list4
            rectange_list=[x for x in result_list if x["image_id"]==images_dict["image id"]]
            len_per_image=len(rectange_list)
            print("len_per_image",len_per_image)
            draw_rectange1(img,savepath,file_name,rectange_list,len_per_image,thickness=10,output_prefix=output_prefix,mode=mode)
        elif not test and (id in imgfilters):
            img=cv2.imread(os.path.join(image_folder_test,file_name))
            rectange_list=[x for x in result_list if x["image_id"]==images_dict["image id"]]
            rectange_list=[x for x in result_list if x["image_id"]==images_dict["image id"]]

            rectange_list1=[x for x in result_list if x["category_id"]==1]
            rectange_list2=[x for x in result_list if x["category_id"]==2]
            rectange_list3=[x for x in result_list if x["category_id"]==3]
            rectange_list4=[x for x in result_list if x["category_id"]==4]
            
            rectange_list1= sorted(rectange_list1, key=itemgetter('score'),reverse=True)[0:500]
            print(len(rectange_list1))
            rectange_list2= sorted(rectange_list2, key=itemgetter('score'),reverse=True)[0:500]
            print(len(rectange_list2))
            rectange_list3= sorted(rectange_list3, key=itemgetter('score'),reverse=True)[0:500]
            print(len(rectange_list3))
            rectange_list4= sorted(rectange_list4, key=itemgetter('score'),reverse=True)[0:500]
            print(len(rectange_list4))
            rectange_list=rectange_list1+rectange_list2+rectange_list3+rectange_list4
            rectange_list=[x for x in rectange_list if x["image_id"]==images_dict["image id"]]
            len_per_image=len(rectange_list)
            draw_rectange1(img,savepath,file_name,rectange_list,len_per_image,thickness=10,output_prefix=output_prefix,mode=mode)
def main():
    """merge----inference"""
    test=1
    imgfilters=["15_27","14_06","16_01","17_23","18_01"]
    image_prefix="det"
    merge_result_visual(image_folder_test="/root/data/gvision/dataset/raw_data/image_test",
            # result_path="/root/data/gvision/final_merge/fusion_results/final_nms_all.json",
            result_path="/root/data/gvision/my_merge/finalsubmission/final2/vehicle.json",
            # result_path="/root/data/gvision/final_merge/final3/fusion_all.json",
            # result_path="/root/data/gvision/final_merge/final2/det_results.json",
            # result_path="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results.json",
            # result_path="/root/data/gvision/final_merge/fusion_results/final_det_det7755_all.json",
            # result_path="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results_4.json",
            # result_path="/root/data/gvision/my_merge/finalsubmission/final2/123_my4.json",
            # result_path="/root/data/gvision/final_merge/fusion_results/final_det_det_all.json",
            # result_path="/root/data/gvision/final_merge/fusion_results/final_4.json",
            annos_path="/root/data/gvision/dataset/raw_data/image_annos/person_bbox_test.json",
            savepath="/root/data/gvision/my_merge/finalsubmission/fafafinal",
            output_prefix=image_prefix,
            imgfilters=imgfilters,
            test=test,
            mode="xywh",
            num=10)#mode: input_bbox mode
if __name__ == "__main__":
    main()

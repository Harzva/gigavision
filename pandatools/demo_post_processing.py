import json
import cv2
import os
import csv
mergeresult="/root/data/gvision/dataset/output/my_pv_inference/my_predict/pre_result.json"
os.makedirs("/root/data/gvision/dataset/output/my_pv_inference/my_merge_anslysis",exist_ok=True)
save_path=os.path.join("/root/data/gvision/dataset/output/my_pv_inference/my_merge_anslysis","pre_result.csv")

def Duplicate_removal(merge_list):
    d_list = []
    merge_list_removed = []
    for dict_item in merge_list:
        val_a = dict_item['image_id']
        val_b = dict_item['category_id']
        val_c = dict_item['bbox']
        val_d = dict_item['score']
        new_tuple = (val_a, val_b,val_c,val_d  )
        if new_tuple not in d_list:
            d_list.append(new_tuple)
            merge_list_removed.append(dict_item)
        # else:
        #     print('the removed element:', dict_item)
    return merge_list_removed 


def post_processing(mergeresult,save_path,mode="pv"):
    """
    data analysis ,coco format save,output visual,megrgeresult visual-----
    """
    merge_result_list = json.load(open(mergeresult,"r"))
    print(merge_result_list[0:5])
    if mode=="person":
        length=5
    if mode=="viechle":
        length=3
    if mode=="pv":
        length=6
    print("DUP removal before:",len(merge_result_list))
    merge_result_list=Duplicate_removal(merge_result_list)

    print("DUP removal after:",len(merge_result_list))
    myList = [([0] * length) for i in range(1)]
    count = [0] * length
    for k in range(len(merge_result_list)):
        image_id = merge_result_list[k]["image_id"]
        category_id = merge_result_list[k]["category_id"]
        bbox = merge_result_list[k]["bbox"]
        score = merge_result_list[k]["score"]

        flag = True
        for i in range(len(myList)):
            if myList[i][0] == image_id:
                flag = False
                myList[i][category_id+1] += 1
                myList[i][length-1] += 1
                count[category_id+1] += 1
                count[length-1] += 1
        if flag == True:
            #print(image_id)
            myList.append([0] * length)
            myList[len(myList)-1][0] = image_id
            myList[len(myList)-1][category_id+1] += 1
            myList[len(myList)-1][length-1] += 1
            count[category_id+1] += 1
            count[length-1] += 1
    myList.pop(0)
    # print(myList)
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["img","0","1","2","3","num"])
        for row in myList:
            writer.writerow(row)
        writer.writerow(count)
def main():
    post_processing(mergeresult,save_path)
if __name__ == "__main__":
    main()






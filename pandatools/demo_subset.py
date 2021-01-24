import json
with open("/root/data/gvision/dataset/image_annos/person_bbox_test.json",'r') as load_f:
    d = json.load(load_f)
# print(d)
del_keys=[]
append_val=[]
val
for key,value in d.items():
    print(key)
    # if "14_OCT_Habour" in key or "15_Nanshani_Park" in key or "17_New_Zhongguan" in key:
    if "18_Xili_Street" in key:
        del_keys.append(key)

    if "16_Primary_School" in key:
        # print(key)
        del_keys.append(key)
    
# print(del_keys)
# for i in del_keys:
#     d.pop(i)
# for i not in del_keys:

# print(d)
with open("/root/data/gvision/dataset/image_annos/person_bbox_test_14_15_17.json",'w') as load_f:
    dict_str=json.dumps(d)
    load_f.write(dict_str)
# print(d)



import json
with open("/root/data/gvision/dataset/image_annos/person_bbox_test.json",'r') as load_f:
    d = json.load(load_f)
# print(d)
del_keys=[]
val={}
for key,value in d.items():
    print(key)
    # if "14_OCT_Habour" in key or "15_Nanshani_Park" in key or "17_New_Zhongguan" in key:
    if "18_Xili_Street" in key:
        del_keys.append(key)
        val["18_Xili_Street"]=value
    if "16_Primary_School" in key:
        # print(key)
        del_keys.append(key)
        val["16_Primary_School"]=value


    
# print(del_keys)
# for i in del_keys:
#     d.pop(i)
# for i not in del_keys:

# print(d)
with open("/root/data/gvision/dataset/image_annos/person_bbox_test_14_15_17.json",'w') as load_f:
    dict_str=json.dumps(d)
    load_f.write(dict_str)
# print(d)
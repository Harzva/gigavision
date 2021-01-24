
import json
import os
import random
# a=[]
# b=[]
# for i in range(391,421):
#     a.append({"image_id": i,"category_id": 4,"bbox": [10832+round(random.uniform(0,1),2),5028++round(random.uniform(0,1),2),1486+round(random.uniform(0,1),2),800+round(random.uniform(0,1),2),],"score": 1.0})
# # print(a)

# for i in range(391,420):
#     b.append({"image_id": i,"category_id": 4,"bbox": [20688+round(random.uniform(0,1),2),4364++round(random.uniform(0,1),2),500+round(random.uniform(0,1),2),744+round(random.uniform(0,1),2),],"score": 1.0})
# # print(b)
# c=a+b
# with open(os.path.join("/root/data/gvision/final_merge","tour.json"), 'w') as f:
#     dict_str = json.dumps(c, indent=2)
#     f.write(dict_str)
#     # print(f'save ***{len(c)} results*** json :{os.path.join("/root/data/gvision/final_merge"，"tour.json")}')
results1="/root/data/gvision/final_merge/head/coco_results/m2retinaface_head.json"
with open(results1, 'r') as load_f:
    model_results= json.load(load_f)
print(model_results[2]["category_id"])
for i in model_results:
    i.update(category_id=3)

with open("/root/data/gvision/final_merge/head/coco_results/m2retinaface_head.json", 'w') as f:
    dict_str = json.dumps(model_results, indent=2)
    f.write(dict_str)
# print(f"save ***results*** json :{os.path.join(outpath, outfile)}")   


import json
import os 
all_path="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results.json"
path_result1="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results_1.json"#17 18
path_result2="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results_2.json"#17 18
path_result3="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results_3.json"
path_result4="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results_4.json"
with open(all_path, 'r') as load_f:
    allresults= json.load(load_f)
result1=[i for i in allresults if i['category_id']==1]
result2=[i for i in allresults if i['category_id']==2]
result3=[i for i in allresults if i['category_id']==3]
result4=[i for i in allresults if i['category_id']==4]
with open(path_result1, 'w') as f:
    dict_str = json.dumps(result1, indent=2)
    f.write(dict_str)
with open(path_result2, 'w') as f:
    dict_str = json.dumps(result2, indent=2)
    f.write(dict_str)
with open(path_result3, 'w') as f:
    dict_str = json.dumps(result3, indent=2)
    f.write(dict_str)
with open(path_result4, 'w') as f:
    dict_str = json.dumps(result4, indent=2)
    f.write(dict_str)


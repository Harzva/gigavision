import pickle
import json
pkl_file = open('/root/data/gvision/dataset/mm_output/my_pv_train/results.pkl','rb')
account_dic = pickle.load(pkl_file)
print(account_dic)
pkl_file.close()
# account_info['349622541'][1] = 18000 　　　
# account_info[''349622541''][2] = 200012 
account_info=account_dic 
# pkl_file=open('/root/data/gvision/dataset/mm_output/my_pv_train/results2.pkl','wb')
# pickle.dump(account_info,pkl_file)
# print(account_info)
# pkl_file.close()
print(len(account_info))
print(type(account_info))
# for i in range(len(account_info)):
#     print(len(account_info[i]))  #44444444444444444
jsonout="/root/data/gvision/dataset/mm_output/my_pv_train/results2.json"
print(f'\nwriting json_results to {jsonout}')
f=open(jsonout, 'w')
result_str = json.dumps(f"{account_info}", indent=2)
f.write(result_str)
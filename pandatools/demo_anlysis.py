# -----------------------train-------------------------------
import json
import csv
import numpy as np

image_id = '390'
mode = 'width'               # mode = width,height,area
category_id = 4              # 4:vehicles

# ----------------------生成csv用,程序开始-----------------------
# with open("D:/giga/Professional/image_annos/coco_person&vehicle.json",'r') as load_f:
#     load_dict = json.load(load_f)
#
# annotations = load_dict["annotations"]
# myList = [([0] * 5) for i in range(1)]
#
# for i in range(len(annotations)):
#     anno = annotations[i]
#     if anno["category_id"] == category_id:
#         # anno["bbox"][2]==w   anno["bbox"][3]==h
#         # image_id = int((anno["image_id"]-1)/30)+1
#         myList.append([anno["image_id"], anno["bbox"][2], anno["bbox"][3], anno["area"]])
#
# myList.pop(0)
#
# with open('C:/Users/Du/Desktop/statistics_vehicles.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["image_id", "width", "height", "area"])
#     for row in myList:
#         writer.writerow(row)
# ----------------------生成csv用,程序结束-----------------------

f = open("C:/Users/Du/Desktop/statistics_vehicles_train.csv", 'r')
final_list = list(csv.reader(f))
list2 = []
for i in range(len(final_list)):
    if final_list[i][0] == image_id:
        list2.append(final_list[i])

if mode == 'width': mode_list = np.array(list2)[:,1]
elif mode == 'height': mode_list = np.array(list2)[:,2]
elif mode == 'area': mode_list = np.array(list2)[:,3]

mode_list = np.core.defchararray.strip(mode_list, '()').astype(float)
print(mode_list)

import matplotlib.pyplot as plt

# matplotlib.axes.Axes.hist() 方法的接口
n, bins, patches = plt.hist(x=mode_list, bins='auto', color='pink',
                            alpha=1, rwidth=0.3)
plt.xticks(bins)
plt.grid(axis='y', alpha=1)
plt.xlabel(mode)
plt.ylabel('Frequency')
plt.title(image_id)
maxfreq = n.max()
# 设置y轴的上限
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.vlines(np.mean(mode_list),0,max(mode_list),colors = "c", linestyles = "dashed")
plt.text(np.mean(mode_list), maxfreq, np.mean(mode_list))
plt.show()



# ---------------------------------------test-------------------------------
# import json
# import csv
# import numpy as np
#
# image_id = '502'
# mode = 'score'               # mode = width,height,area,score
# category_id = 4              # 4:vehicles
#
# # ----------------------生成csv用,程序开始-----------------------
# # with open("D:/giga/Professional/image_annos/nms0.9_ms_0017574_merge_predict_all.json",'r') as load_f:
# #     load_dict = json.load(load_f)
# # annotations = load_dict
# # myList = [([0] * 5) for i in range(1)]
# #
# # for i in range(len(annotations)):
# #     anno = annotations[i]
# #     if anno["category_id"] == category_id:
# #         # anno["bbox"][2]==w   anno["bbox"][3]==h
# #         # image_id = int((anno["image_id"]-1)/30)+1
# #         myList.append([anno["image_id"], anno["bbox"][2], anno["bbox"][3], anno["bbox"][2]*anno["bbox"][3], anno["score"]])
# # myList.pop(0)
# #
# # with open('C:/Users/Du/Desktop/statistics_vehicles_test.csv', 'w', newline='') as csvfile:
# #     writer = csv.writer(csvfile)
# #     writer.writerow(["image_id", "width", "height", "area","score"])
# #     for row in myList:
# #         writer.writerow(row)
# # ----------------------生成csv用,程序结束-----------------------
#
# f = open("C:/Users/Du/Desktop/statistics_vehicles_test.csv", 'r')
# final_list = list(csv.reader(f))
# list2 = []
# for i in range(len(final_list)):
#     if final_list[i][0] == image_id:
#         list2.append(final_list[i])
#
# if mode == 'width': mode_list = np.array(list2)[:,1]
# elif mode == 'height': mode_list = np.array(list2)[:,2]
# elif mode == 'area': mode_list = np.array(list2)[:,3]
# elif mode == 'score': mode_list = np.array(list2)[:,4]
#
# mode_list = np.core.defchararray.strip(mode_list, '()').astype(float)
# print(mode_list)
#
# import matplotlib.pyplot as plt
#
# # matplotlib.axes.Axes.hist() 方法的接口
# n, bins, patches = plt.hist(x=mode_list, bins='auto', color='pink',
#                             alpha=1, rwidth=0.3)
# plt.xticks(bins)
# plt.grid(axis='y', alpha=1)
# plt.xlabel(mode)
# plt.ylabel('Frequency')
# plt.title(image_id)
# maxfreq = n.max()
# # 设置y轴的上限
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.vlines(np.mean(mode_list),0,max(mode_list),colors = "c", linestyles = "dashed")
# plt.text(np.mean(mode_list), maxfreq, np.mean(mode_list))
# plt.show()
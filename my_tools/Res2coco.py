import json
import csv
def res2coco():
    with open("D:/giga/split/det_result.json",'r') as load_f:
        load_dict = json.load(load_f)
    print(len(load_dict))

    myList = [([0] * 5) for i in range(1)]
    count = [0,0,0,0,0]

    for k in range(len(load_dict)):

        image_id = load_dict[k]["image_id"]
        category_id = load_dict[k]["category_id"]
        bbox = load_dict[k]["bbox"]
        score = load_dict[k] = load_dict[k]["score"]

        flag = True
        for i in range(len(myList)):
            if myList[i][0] == image_id:
                flag = False
                myList[i][category_id] += 1
                myList[i][4] += 1
                count[category_id] += 1
                count[4] += 1
        if flag == True:
            myList.append([0] * 5)
            myList[len(myList) - 1][0] = image_id
            myList[i][category_id] += 1
            myList[i][4] += 1

    myList.pop(0)

    with open('D:/giga/split/det_result2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["图片", "1", "2", "3", "总数"])
        for row in myList:
            writer.writerow(row)
        writer.writerow(count)
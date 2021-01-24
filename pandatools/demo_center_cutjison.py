import __future__
import cv2
import random
import os
import numpy as np
name_length = 6
pwd = os.getcwd() # 目录下有中文目录，打印时遇到乱码

def get_data(region_size, start_number):
    path_images = "/Users/xduyzy/Downloads/PANDA_IMAGE/image_train/"
    path_labels = 'image_train/image_annos/'
    save_image_path = 'result/images/1/'
    save_label_path = 'result/annos/1/'

    for image_name in os.listdir(path_images):
        label_name = image_name[:-4] + '.txt'
        image_path = os.path.join(path_images, image_name)
        label_path = os.path.join(path_labels, label_name)
        img = cv2.imread(image_path)
        label = open(label_path,"r")
        lines = label.readlines()
        line_record = []
        for i in range(len(lines)):
            if i in line_record:
                continue
            line_ = lines[i].strip('\r\n')
            strline_ = (((line_.replace('(', '')).replace(')', '')).replace('(', '')).replace(')', '')
            strline = strline_.split(",")
            x1 = float(strline[0])
            y1 = float(strline[1])
            width = float(strline[2])
            height = float(strline[3])
            confidence = int(float(strline[4]))
            if confidence != 1:
                continue
            cate = int(float(strline[5]))
            # if cate==0 or cate==11:
            #    continue
            center_x = width/2+x1
            center_y = height/2+y1

            if width<=3:
                continue
            if height<=3:
                continue

            rand_int = float(random.uniform(-region_size[0]//4, region_size[1]//4))
            # + rand_int
            left = int(center_x - region_size[1]//2 + rand_int)
            right = int(center_x + region_size[1]//2 + rand_int)
            upper = int(center_y - region_size[0]//2 + rand_int)
            bottom = int(center_y + region_size[0]//2 + rand_int)

            if left<=0:
                left = 1
                right = region_size[1]
            if right >= img.shape[1]:
                left = img.shape[1]-region_size[1]
                right = img.shape[1]
            if upper<=0:
                upper = 1
                bottom = region_size[0]
            if bottom >= img.shape[0]:
                upper = img.shape[0]-region_size[0]
                bottom = img.shape[0]

            img_region = img[upper:bottom,left:right,:]

            label_list = []
            for j in range(len(lines)):
                line_1 = lines[j].strip('\r\n')
                strline_1 = (((line_1.replace('(', '')).replace(')', '')).replace('(', '')).replace(')', '')
                strline1 = strline_1.split(",")
                x11 = float(strline1[0])
                y11 = float(strline1[1])
                w = float(strline1[2])
                h = float(strline1[3])
                confidence1 = int(float(strline1[4]))
                if confidence1 != 1:
                    continue
                c = int(float(strline1[5]))
                truincation = float(strline1[6])
                occlusion = float(strline1[7])
                # if c == 0 or c == 11:
                #    continue
                c_x = w/2+x11
                c_y = h/2+y11
                if w <= 3:
                    continue
                if h <= 3:
                    continue
                tt = 1
                if c_x>=left+w*0.2 and c_x<=right-w*0.2 and c_y>=upper+h*0.2 and c_y<=bottom-h*0.2:
                    b_left = c_x - w/2 - left
                    b_upper = c_y - h/2 - upper
                    b_right = c_x + w/2 - left
                    b_bottom = c_y + h/2 - upper
                    if b_left <= 0:
                        b_left = 1
                        tt = 0
                    if b_upper <= 0:
                        b_upper = 1
                        tt = 0
                    if b_right >= region_size[1]:
                        b_right = region_size[1]
                        tt = 0
                    if b_bottom >= region_size[0]:
                        b_bottom = region_size[0]
                        tt = 0
                    if tt==1:
                        line_record.append(j)
                    b_width =  b_right - b_left
                    b_height = b_bottom - b_upper
                    label_list.append((int(b_left), int(b_upper), int(b_width), int(b_height), int(confidence1), int(c), truincation, occlusion))
            new_name_str = (name_length - len(str(start_number))) * '0' + str(start_number)
            with open(save_label_path+ new_name_str+".txt","w") as f:
                for m in range(len(label_list)):
                    f.write(str(label_list[m][0])+","+str(label_list[m][1])+","+str(label_list[m][2])+","+str(label_list[m][3])+","+str(label_list[m][4])+ "," + str(label_list[m][5])+"," + str(label_list[m][6])+"," + str(label_list[m][7]) + '\r\n')
                f.close()
            cv2.imwrite(save_image_path + new_name_str + ".jpg", img_region)
            start_number = start_number + 1
            print(start_number)
    return start_number

if __name__ == "__main__":
    order = get_data((256, 256), 0)
    # order400 = get_data(400, order)
    # order600 = get_data(600, order400)
    # print order600,order600-order
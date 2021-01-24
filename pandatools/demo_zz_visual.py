import os
import json
import glob
import random
import cv2


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])
def loadImg(imgpath):
    """
    :param imgpath: the path of image to load
    :return: loaded img object
    """
    #print('filename:', imgpath)
    if not os.path.exists(imgpath):
        print('Can not find {}, please check local dataset!'.format(imgpath))
        return None
    img = cv2.imread(imgpath)
    imgheight, imgwidth = img.shape[:2]
    scale = 1
    img = cv2.resize(img, (int(imgwidth * scale), int(imgheight * scale)))

    return img
def annos_visual()
    tgtfile = "C:/Users/Du/Desktop/demo2.json"
    savedir = "D:/giga/result/"
    with open(tgtfile, 'r') as load_f:
        anno_dict = json.load(load_f)

    allnames = list(anno_dict['images'])
    imgnames=[]

    for a in allnames:
        dic=dict()
        dic["name"]=a["file_name"]
        dic["id"] = a["id"]
        imgnames.append(dic)

    imagepath = "D:/giga/Professional/image_test/"
    for item in imgnames:
        imgname= item['name']
        imgid=item['id']
        imgpath = os.path.join(imagepath, imgname)
        img = loadImg(imgpath)
        if img is None:
            continue
        for objdict in anno_dict["annotations"]:

            if objdict["image_id"] == imgid:
                print(objdict)
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)
                xmin, ymin, w, h = objdict["bbox"]
                xmax = xmin + w
                ymax = ymin + h
                print('准备画框')
                img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (b, g, r), 8,lineType=8))
                cv2.putText(img, '{}'.format(objdict["categroy_id"]), (50,150), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 12)

                #img = cv2.rectangle(img, (w, h),(w+xmin, h+ymin),  (b, g, r), 3)
        print('准备保存')
        cv2.imwrite(os.path.join(savedir, custombasename(imgname) + '.jpg'), img)

def main():
    annos_visual()
if __name__ == "__main__":
    main()

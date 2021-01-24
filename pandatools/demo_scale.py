import cv2
for i in range(10):
    scale=i*0.1+0.1
    img=r"/root/data/gvision/dataset/image_train/01_University_Canteen/IMG_01_01.jpg"
    img = cv2.imread(img,-1)  
    imgheight, imgwidth = img.shape[:2]
    print(imgheight, imgwidth )
    resizeimg = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/root/data/gvision/dataset/zip/scale/s{}_IMG_01_01.jpg".format(scale),resizeimg)
    imgheight, imgwidth = resizeimg.shape[:2]
    print(imgheight, imgwidth)

"""
# 缩小图像  
size = (int(width*0.3), int(height*0.5))  
shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  
    
# 放大图像  
fx = 1.6  
fy = 1.2  
enlarge = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)  
"""

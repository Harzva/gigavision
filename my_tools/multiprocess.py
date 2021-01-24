# --------------------------------------------------------
# Image and annotations splitting modules for PANDA
# Written by Wang Xueyang  (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

import os
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import json
import copy
from collections import defaultdict


class ImgSplit():
    def __init__(self,
                 basepath,
                 annofile,
                 annomode,
                 outpath,
                 outannofile,
                 code='utf-8',
                 outimage='image_train',
                 gap=(100, 100),
                 subwidth=2048,
                 subheight=1024,
                 thresh=0.7,
                 outext='.jpg',
                 ):
        """
        :param basepath: base directory for panda image data and annotations
        :param annofile: annotation file path
        :param annomode:the type of annotation, which can be 'person', 'vehicle', 'headbbox' or 'headpoint'
        :param outpath: output base path for panda data
        :param outannofile: output file path for annotation
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subwidth: sub-width of patch
        :param subheight: sub-height of patch
        :param thresh: the square thresh determine whether to keep the instance which is cut in the process of split
        :param outext: ext for the output image format
        """
        self.basepath = basepath
        self.annofile = annofile
        self.annomode = annomode
        self.outpath = outpath
        self.outannofile = outannofile
        self.code = code
        self.gap = gap
        self.subwidth = subwidth
        self.subheight = subheight
        self.slidewidth = self.subwidth - self.gap[0]
        self.slideheight = self.subheight - self.gap[1]
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, outimage)
        self.annopath = os.path.join(self.basepath, 'image_annos', annofile)
        self.outimagepath = os.path.join(self.outpath, outimage)
        self.outannopath = os.path.join(self.outpath, 'image_annos')
        self.outext = outext
        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outannopath):
            os.makedirs(self.outannopath)
        self.annos = defaultdict(list)
        self.loadAnno()

    def loadAnno(self):
        print('Loading annotation json file: {}'.format(self.annopath))
        with open(self.annopath, 'r') as load_f:
            annodict = json.load(load_f)
        self.annos = annodict

    def splitdata(self, scale, imgrequest=None, imgfilters=[]):
        """
        :param scale: resize rate before cut
        :param imgrequest: list, images names you want to request, eg. ['1-HIT_canteen/IMG_1_4.jpg', ...]
        :param imgfilters: essential keywords in image name
        """
        if imgrequest is None or not isinstance(imgrequest, list):
            imgnames = list(self.annos.keys())
        else:
            imgnames = imgrequest

        splitannos = {}
        img_list = []
        for imgname in imgnames:
            iskeep = False
            for imgfilter in imgfilters:
                if imgfilter in imgname:
                    iskeep = True
            if imgfilters and not iskeep:
                continue
            img_list.append(imgname)

        def split_img(imgname):
            splitdict = self.SplitSingle(imgname, scale)
            return splitdict

        executor = ThreadPoolExecutor(max_workers=10)
        for splitdict in executor.map(split_img, img_list):
            # for img in img_list:
            #     splitdict = split_img(img)
            splitannos.update(splitdict)
        # add image id
        imgid = 1
        for imagename in splitannos.keys():
            splitannos[imagename]['image id'] = imgid
            imgid += 1
        # save new annotation for split images
        outdir = os.path.join(self.outannopath, self.outannofile)
        with open(outdir, 'w', encoding=self.code) as f:
            dict_str = json.dumps(splitannos, indent=2)
            f.write(dict_str)

    def splitdata_test(self, scale, imgrequest=None, imgfilters=[]):
        """
        :param scale: resize rate before cut
        :param imgrequest: list, images names you want to request, eg. ['1-HIT_canteen/IMG_1_4.jpg', ...]
        :param imgfilters: essential keywords in image name
        """
        if imgrequest is None or not isinstance(imgrequest, list):
            imgnames = list(self.annos.keys())
        else:
            imgnames = imgrequest

        splitannos = {}
        img_list = []
        for imgname in imgnames:
            iskeep = False
            for imgfilter in imgfilters:
                if imgfilter in imgname:
                    iskeep = True
            if imgfilters and not iskeep:
                continue
            img_list.append(imgname)

        def split_img(imgname):
            splitdict = self.SplitSingle_test(imgname, scale)
            return splitdict

        executor = ThreadPoolExecutor(max_workers=10)
        for splitdict in executor.map(split_img, img_list):
            # for img in img_list:
            #     splitdict = split_img(img)
            splitannos.update(splitdict)
        # add image id
        imgid = 1
        for imagename in splitannos.keys():
            splitannos[imagename]['image id'] = imgid
            imgid += 1
        # save new annotation for split images
        outdir = os.path.join(self.outannopath, self.outannofile)
        with open(outdir, 'w', encoding=self.code) as f:
            dict_str = json.dumps(splitannos, indent=2)
            f.write(dict_str)

    def loadImg(self, imgpath):
        """
        :param imgpath: the path of image to load
        :return: loaded img object
        """
        print('filename:', imgpath)
        if not os.path.exists(imgpath):
            print('Can not find {}, please check local dataset!'.format(imgpath))
            return None
        img = cv2.imread(imgpath)
        return img

    def SplitSingle(self, imgname, scale):
        """
        split a single image and ground truth
        :param imgname: image name
        :param scale: the resize scale for the image
        :return:
        """
        imgpath = os.path.join(self.imagepath, imgname)
        img = self.loadImg(imgpath)
        if img is None:
            return
        imagedict = self.annos[imgname]
        objlist = imagedict['objects list']

        # re-scale image if scale != 1
        if scale != 1:
            resizeimg = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        imgheight, imgwidth = resizeimg.shape[:2]

        # split image and annotation in sliding window manner
        outbasename = imgname.replace('/', '_').split('.')[0] + '___' + str(scale) + '__'
        subimageannos = {}
        left, up = 0, 0
        while left < imgwidth:
            if left + self.subwidth >= imgwidth:
                left = max(imgwidth - self.subwidth, 0)
            up = 0
            while up < imgheight:
                if up + self.subheight >= imgheight:
                    up = max(imgheight - self.subheight, 0)
                right = min(left + self.subwidth, imgwidth - 1)
                down = min(up + self.subheight, imgheight - 1)
                coordinates = left, up, right, down
                subimgname = outbasename + str(left) + '__' + str(up) + self.outext
                self.savesubimage(resizeimg, subimgname, coordinates)
                # split annotations according to annotation mode
                if self.annomode == 'person':
                    newobjlist = self.personAnnoSplit(objlist, imgwidth, imgheight, coordinates)
                elif self.annomode == 'vehicle':
                    newobjlist = self.vehicleAnnoSplit(objlist, imgwidth, imgheight, coordinates)
                subimageannos[subimgname] = {
                    "image size": {
                        "height": down - up + 1,
                        "width": right - left + 1
                    },
                    "objects list": newobjlist
                }
                if up + self.subheight >= imgheight:
                    break
                else:
                    up = up + self.slideheight
            if left + self.subwidth >= imgwidth:
                break
            else:
                left = left + self.slidewidth

        return subimageannos

    @staticmethod
    def roi_int(rectdict, imgwidth, imgheight, coordinates, only_roi=False):
        left, up, right, down = coordinates
        xmin = int(rectdict['tl']['x'] * imgwidth)
        ymin = int(rectdict['tl']['y'] * imgheight)
        xmax = int(rectdict['br']['x'] * imgwidth)
        ymax = int(rectdict['br']['y'] * imgheight)
        square = (xmax - xmin) * (ymax - ymin)

        if (xmax <= left or right <= xmin) and (ymax <= up or down <= ymin):
            intersection = 0
        else:
            lens = min(xmax, right) - max(xmin, left)
            wide = min(ymax, down) - max(ymin, up)
            intersection = lens * wide
        if only_roi:
            return intersection / (square + 1e-5)
        return intersection, intersection / (square + 1e-5)

    def judgeRect(self, rectdict, imgwidth, imgheight, coordinates):
        intersection, roi = self.roi_int(rectdict, imgwidth, imgheight, coordinates)
        return intersection and roi > self.thresh

    def restrainRect(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        xmin = int(rectdict['tl']['x'] * imgwidth)
        ymin = int(rectdict['tl']['y'] * imgheight)
        xmax = int(rectdict['br']['x'] * imgwidth)
        ymax = int(rectdict['br']['y'] * imgheight)
        xmin = max(xmin, left)
        xmax = min(xmax, right)
        ymin = max(ymin, up)
        ymax = min(ymax, down)
        return {
            'tl': {
                'x': (xmin - left) / (right - left),
                'y': (ymin - up) / (down - up)
            },
            'br': {
                'x': (xmax - left) / (right - left),
                'y': (ymax - up) / (down - up)
            }
        }

    def judgePoint(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        x = int(rectdict['x'] * imgwidth)
        y = int(rectdict['y'] * imgheight)

        if left < x < right and up < y < down:
            return True
        else:
            return False

    def restrainPoint(self, rectdict, imgwidth, imgheight, coordinates):
        left, up, right, down = coordinates
        x = int(rectdict['x'] * imgwidth)
        y = int(rectdict['y'] * imgheight)
        return {
            'x': (x - left) / (right - left),
            'y': (y - up) / (down - up)
        }

    def personAnnoSplit(self, objlist, imgwidth, imgheight, coordinates):
        newobjlist = []
        for object_dict in objlist:
            objcate = object_dict['category']
            if objcate == 'person':
                pose = object_dict['pose']
                riding = object_dict['riding type']
                age = object_dict['age']
                fullrect = object_dict['rects']['full body']
                visiblerect = object_dict['rects']['visible body']
                headrect = object_dict['rects']['head']
                # only keep a person whose 3 box all satisfy the requirement
                if self.judgeRect(fullrect, imgwidth, imgheight, coordinates) & \
                        self.judgeRect(visiblerect, imgwidth, imgheight, coordinates) & \
                        self.judgeRect(headrect, imgwidth, imgheight, coordinates):
                    newobjlist.append({
                        "category": objcate,
                        "pose": pose,
                        "riding type": riding,
                        "age": age,
                        "ignore": 0,
                        "rects": {
                            "head": self.restrainRect(headrect, imgwidth, imgheight, coordinates),
                            "visible body": self.restrainRect(visiblerect, imgwidth, imgheight, coordinates),
                            "full body": self.restrainRect(fullrect, imgwidth, imgheight, coordinates)
                        }
                    })
                elif self.judgeRect(visiblerect, imgwidth, imgheight, coordinates):
                    newobjlist.append({
                        "category": objcate,
                        "pose": pose,
                        "riding type": riding,
                        "age": age,
                        "ignore": 1,
                        "rects": {
                            "head": self.restrainRect(headrect, imgwidth, imgheight, coordinates),
                            "visible body": self.restrainRect(visiblerect, imgwidth, imgheight, coordinates),
                            "full body": self.restrainRect(fullrect, imgwidth, imgheight, coordinates)
                        }
                    })
            else:
                rect = object_dict['rect']
                if self.roi_int(rect, imgwidth, imgheight, coordinates, only_roi=True) > 0.65:
                    newobjlist.append({
                        "category": objcate,
                        "ignore": 1,
                        "rect": self.restrainRect(rect, imgwidth, imgheight, coordinates)
                    })

        return newobjlist

    def vehicleAnnoSplit(self, objlist, imgwidth, imgheight, coordinates):
        newobjlist = []
        for object_dict in objlist:
            objcate = object_dict['category']
            rect = object_dict['rect']
            if self.judgeRect(rect, imgwidth, imgheight, coordinates):
                newobjlist.append({
                    "category": objcate,
                    "ignore": 0,
                    "rect": self.restrainRect(rect, imgwidth, imgheight, coordinates)
                })
            elif self.roi_int(rect, imgwidth, imgheight, coordinates, only_roi=True) > 0.6:
                newobjlist.append({
                    "category": objcate,
                    "ignore": 1,
                    "rect": self.restrainRect(rect, imgwidth, imgheight, coordinates)
                })
        return newobjlist

    def savesubimage(self, img, subimgname, coordinates):
        left, up, right, down = coordinates
        subimg = copy.deepcopy(img[up: down, left: right])
        outdir = os.path.join(self.outimagepath, subimgname)
        cv2.imwrite(outdir, subimg)

    def SplitSingle_test(self, imgname, scale):
        """
        split a single image and ground truth
        :param imgname: image name
        :param scale: the resize scale for the image
        :return:
        """
        imgpath = os.path.join(self.imagepath, imgname)
        img = self.loadImg(imgpath)
        if img is None:
            return
        imagedict = self.annos[imgname]

        # re-scale image if scale != 1
        if scale != 1:
            resizeimg = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        imgheight, imgwidth = resizeimg.shape[:2]

        # split image and annotation in sliding window manner
        outbasename = imgname.replace('/', '_').split('.')[0] + '___' + str(scale) + '__'
        subimageannos = {}
        left, up = 0, 0
        while left < imgwidth:
            if left + self.subwidth >= imgwidth:
                left = max(imgwidth - self.subwidth, 0)
            up = 0
            while up < imgheight:
                if up + self.subheight >= imgheight:
                    up = max(imgheight - self.subheight, 0)
                right = min(left + self.subwidth, imgwidth - 1)
                down = min(up + self.subheight, imgheight - 1)
                coordinates = left, up, right, down
                subimgname = outbasename + str(left) + '__' + str(up) + self.outext
                self.savesubimage(resizeimg, subimgname, coordinates)
                # split annotations according to annotation mode

                subimageannos[subimgname] = {
                    "image size": {
                        "height": down - up + 1,
                        "width": right - left + 1
                    },
                }
                if up + self.subheight >= imgheight:
                    break
                else:
                    up = up + self.slideheight
            if left + self.subwidth >= imgwidth:
                break
            else:
                left = left + self.slidewidth

        return subimageannos
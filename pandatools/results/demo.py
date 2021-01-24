# --------------------------------------------------------
# Tool kit function demonstration
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200321
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ImgSplit import ImgSplit
from ResultMerge import DetResMerge

if __name__ == '__main__':
    image_root = 'G:/PANDA/PANDA_IMAGE'
    person_anno_file = 'human_bbox_train.json'
    annomode = 'person'
    example = PANDA_IMAGE(image_root, person_anno_file, annomode='person')

    '''1. show images'''
    example.showImgs()

    '''2. show annotations'''
    example.showAnns(range=50, shuffle=True)

    '''
    3. Split Image And Label
    We provide the scale param before split the images and labels.
    '''
    outpath = 'split'
    outannofile = 'split.json'
    split = ImgSplit(image_root, person_anno_file, annomode, outpath, outannofile)
    split.splitdata(0.5)

    '''
    4. Merge patches
    Now, we will merge these patches to see if they can be restored in the initial large images
    '''
    GT2DetRes(gtpath, outdetpath):
    util.GT2DetRes('split/annoJSONs/split.json', 'split/resJSONs/res.json')
        """
        transfer format: groundtruth to detection results
    :param gtpath: the path to groundtruth json file
    :param outdetpath:the path to output detection result json file
    :return:
    """
        self.respath = os.path.join(self.basepath, 'resJSONs', resfile)
        self.splitannopath = os.path.join(self.basepath, 'image_annos', splitannofile)
        self.srcannopath = os.path.join(self.basepath, 'image_annos', srcannofile)
    # merge = DetResMerge('dataset/split', 'res.json', 'split.json', 'human_bbox_all.json', 'results', 'mergetest.json')
    basepath='dataset/split/person_s0.5_t0.7_01_02'
    resfile="split_result.json"#检测结果未融合
    splitannofile="coco_person_bbox_split_02_train.json"#切图的输入结果
    srcannofile="dataset/image_annos/person_bbox_train.json"#未切图的结果
    outpath="dataset/split/person_s0.5_t0.7_01_02/resJSONS"
    outfile="merge_result.json"#融合的检测结果
    merge = DetResMerge(basepath, resfile, splitannofile, srcannofile, outpath, outfile)
    merge.mergeResults(is_nms=False)
               """
        :param basepath: base directory for panda image data and annotations
        :param resfile: detection result file path
        :param splitannofile: generated split annotation file
        :param srcannofile: source annotation file
        :param outpath: output base path for merged result file
        :param outfile: name for merged result file
        :param imgext: ext for the split image format
        """
        DetRes2GT(detrespath, outgtpath, gtannopath):
    util.DetRes2GT('results/mergetest.json', 'G:/annoJSONs/mergegt.json', 'G:/annoJSONs/human_bbox_all.json')
        """
        transfer format: detection results to groundtruth
    :param detrespath: the path to input detection result json file
    :param outgtpath: the path to output groundtruth json file
    :param gtannopath: source annotation json file path for image data
    :return:
    """

    '''show merged results'''
    example = PANDA_IMAGE(image_root, 'mergegt.json', annomode='vehicle')
    example.showAnns()

    '''5. PANDA video visualization test'''
    video_root = 'G:/PANDA/PANDA_VIDEO'
    video_savepath = 'results'
    request = ['02_OCT_Habour']
    example = PANDA_VIDEO(video_root, video_savepath)

    '''save video'''
    example.saveVideo(videorequest=request, maxframe=100)

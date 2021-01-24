"""
This is a Python version used to implement the Soft NMS algorithm.
Original Paper：Improving Object Detection With One Line of Code
"""
import numpy as np
def py_cpu_softnms(dets,Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """
    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, 4]
    # scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep
def set_cpu_nms(dets, thresh,use="merge"):
        """
        [829, 5939, 923, 6000, 0.24672751128673553, 1, 149]
        :dets: 二维numpy.ndarray, 每行6列 [x,y,w,h,score,cat,number],需要保证在同一个set里的boxes的number是唯一的
        :return: bool 型numpy.ndarray, the index of keepded boxes.
        """
        def _overlap(det_boxes, basement, others):
            eps = 1e-8
            x1_basement, y1_basement, x2_basement, y2_basement \
                    = det_boxes[basement, 0], det_boxes[basement, 1], \
                    det_boxes[basement, 2], det_boxes[basement, 3]
            x1_others, y1_others, x2_others, y2_others \
                    = det_boxes[others, 0], det_boxes[others, 1], \
                    det_boxes[others, 2], det_boxes[others, 3]
            areas_basement = (x2_basement - x1_basement) * (y2_basement - y1_basement)
            areas_others = (x2_others - x1_others) * (y2_others - y1_others)
            xx1 = np.maximum(x1_basement, x1_others)
            yy1 = np.maximum(y1_basement, y1_others)
            xx2 = np.minimum(x2_basement, x2_others)
            yy2 = np.minimum(y2_basement, y2_others)
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas_basement + areas_others - inter + eps)
            return ovr
        scores = dets[:, 4]
        order = np.argsort(-scores)
        dets = dets[order] #按score从大到小排序
        #change to l t r d
        if use=="model":
            dets[:,2] = dets[:,2]+dets[:,0]
            dets[:,3] = dets[:,3]+dets[:,1]
        if use=="merge":
            dets[:,2] = dets[:,2]
            dets[:,3] = dets[:,3]


        numbers = dets[:, -1]  #  set number
        print("numbers",numbers)
        keep = np.ones(len(dets)) == 1 # keep all at begining
        ruler = np.arange(len(dets)) # ruler = index of order # [0,1,2,3,4.....len]
        while ruler.size>0:
            basement = ruler[0]
            ruler=ruler[1:]
            num = numbers[basement]
            # calculate the body overlap
            overlap = _overlap(dets[:, :4], basement, ruler)
            indices = np.where(overlap > thresh)[0] 
            loc = np.where(numbers[ruler][indices] == num)[0] 
            # the mask won't change in the step
            mask = keep[ruler[indices][loc]]
            keep[ruler[indices]] = False
            keep[ruler[indices][loc][mask]] = True
            ruler[~keep[ruler]] = -1
            ruler = ruler[ruler>0]
        # print("np.argsort(order)",np.argsort(order))#438
        # print("np.argsort(order)--len",len(np.argsort(order)))
        # print("keep[np.argsort(order)]--len",len(keep[np.argsort(order)]))
        # print("keep[np.argsort(order)]",keep[np.argsort(order)])#438
        # keep = np.argsort(order)# 438 数字
        keep = keep[np.argsort(order)]#false
        print(keep)

        return keep

def py_cpu_nms(dets, thresh):
    # print('dets:', len(dets))
    # print(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]+dets[:, 0]
    y2 = dets[:, 3]+dets[:, 1]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # print("lenareas",len(areas))
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    # print("keep",keep)

    return keep
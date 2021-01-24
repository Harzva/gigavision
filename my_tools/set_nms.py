import numpy as np

def set_cpu_nms(self, dets, thresh):
        """
        :dets: 二维numpy.ndarray, 每行6列 [x,y,w,h,score,number],需要保证在同一个set里的boxes的number是唯一的
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
        dets[:,2] = dets[:,2]+dets[:,0]
        dets[:,3] = dets[:,3]+dets[:,1]

        numbers = dets[:, -1]  #  set number
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
        keep = keep[np.argsort(order)]
        return keep
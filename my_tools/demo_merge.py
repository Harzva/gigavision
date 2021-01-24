import ResultMerge
def merge():
    print("--------->>>>>>>>>merge-------------start")
    merge =ResultMerge.DetResMerge(resfile="/root/data/gvision/dataset/rs_output/my_p/results.json.bbox.json", 
                            splitannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json" ,
                            srcannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517.json",
                            outpath="/root/data/gvision/dataset/rs_output/my_p",
                            )
    nms_thresh_list=[0.9,0.8]        
    for i in nms_thresh_list:
        merge.mergeResults(is_nms=True,nms_thresh=i,outfile=f"my_predict/nms{i}_merge_predict.json")

if __name__ == "__main__":
    merge()

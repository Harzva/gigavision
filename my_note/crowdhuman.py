python tools/convert_datasets/crowd_human.py --path /home/ubuntu/Documents/hzh/gigavision/data/crowd_human --kind full

python tools/test.py configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py work_dirs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x/crowd_human_full_faster_rcnn_r50_fpn_2x.pth --out result.pkl --eval bbox



CUDA_VISIBLE_DEVICES=0,1,2,4 PORT=29501 ./tools/dist_test.sh 	configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py \
	work_dirs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x/crowd_human_full_faster_rcnn_r50_fpn_2x.pth  4 \
	--out /home/ubuntu/Documents/hzh/gigavision/BAAI-2020-CrowdHuman-Baseline/work_dirs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x/det_results.pkl

CUDA_VISIBLE_DEVICES=0,1,2,4 PORT=29501 ./tools/dist_test.sh 	configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py \
	work_dirs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x/crowd_human_full_faster_rcnn_r50_fpn_2x.pth  4 \
	--out /home/ubuntu/Documents/hzh/gigavision/BAAI-2020-CrowdHuman-Baseline/work_dirs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x/results.pkl --eval bbox



    3 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_fullbody_method2/output/detectors_epoch_30_fafaxue" \
    --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_fullbody_method2/visual/fafaxue 	

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
	vim ~/.bashrc 
	
	sudo ldconfig /usr/local/cuda-10.0/lib64
	
	
	conda create -n NEW --clone OLD 

{'gts': 99481, 'dets': 355082, 'recall': 0.947628190307697, 'mAP': 0.8931054528057181, 'mMR': 0.47554132576820124}
------------------------------------------------------------------------------------------------------eval


python test.py -md rcnn_emd_refine -r /home/ubuntu/Documents/hzh/gigavision/CrowdDet-master/pretrained/rcnn_emd_refine_mge.pth  -d 0-1-2 
-------------------------------------------------------------------test

python mytest.py -md rcnn_emd_refine -r /home/ubuntu/Documents/hzh/gigavision/CrowdDet-master/model/rcnn_emd_refine/outputs_init/model_dump/dump-30.pth -d 0-1-2-4


 python visulize_json.py -f /home/ubuntu/Documents/hzh/gigavision/CrowdDet-master/model/rcnn_emd_refine/outputs/result_dump/dump-nms0.5prethre0.7_rcnn_emd_refine_mge.json
--------------------------train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -md rcnn_emd_refine -r 30



python visulize_json.py -f /home/ubuntu/Documents/hzh/gigavision/Pedestron/work_dirs/crowdhuman_cascade_rcnn_hrnet/result_humanrowd0.txt -n 11
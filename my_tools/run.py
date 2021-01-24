


10.170.23.49
实际任务一定不是按这样划分的呀，不是应该每个场景都给一些带有标签信息的图片，然后18个场景进行预测吗，固定的摄像头就会有固定的背景信息，实际生活种
同样是标数据，最好的方法就是每个场景都标几张，然后训练，最后预测这些固定的场景。
pyramidbox
python /root/data/gvision/head_model/2018--ZJUAI--PyramidBoxDetector-master/test_yuncong_our_test2.py \
--resume '/root/data/gvision/head_model/2018--ZJUAI--PyramidBoxDetector-master/weights/best_mod_Res50_pyramid.pth'
python /root/data/gvision/head_model/2018--ZJUAI--PyramidBoxDetector-master/test_yuncong_our_test2.py \
--resume '/root/data/gvision/head_model/2018--ZJUAI--PyramidBoxDetector-master/weights/best_our_Res50_pyramid_aug.pth'

visuasl
 python visualize.py json/20200707_005329.log.json 
python tools/voc_eval_visualize.py result.pkl ./configs/faster_rcnn_r101_fpn_1x.py
-----------------------------------D2det
python tools/test.py /root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_visbilebody_test_noms.py \
/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/net_output/epoch_8.pth \
--out /root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_noms.pkl \
--format_only \
--options jsonfile_prefix="/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_noms" 

python tools/test.py /root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_visbilebody_test_noms.py \
/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/net_output/epoch_8.pth \
--out /root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_noms.pkl \
--format_only \
--options jsonfile_prefix="/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_noms" 
CUDA_VISIBLE_DEVICES=1 /root/data/gvision/D2Det-master/tools/dist_test.sh \
/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/net_output/epoch_8.pth \
--format_only \
--options jsonfile_prefix="/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_dyy" 

CUDA_VISIBLE_DEVICES=1 ./root/data/gvision/D2Det-master/tools/dist_test.sh \
	/root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_visbilebody.py  \
    /root/data/gvision/D2Det-master/workdir/otput/D2Det_detection_r101_fpn_dcn_2x_visiblebody/epoch_8.pth \
    1 --format_only --options "jsonfile_prefix=/root/data/gvision/D2Det-master/workdir/otput/D2Det_detection_r101_fpn_dcn_2x_visiblebody/outputs_results_dyy"

CUDA_VISIBLE_DEVICES=1 ./root/data/gvision/D2Det-master/tools/dist_test.sh \
	/root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_vehicle.py \
    /root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_vehicle/latest.pth \
    1 --format_only --options "jsonfile_prefix=/root/data/gvision/D2Det-master/workdir/otput/D2Det_detection_r101_fpn_dcn_2x_vehicle/outputs_results_dyy"

CUDA_VISIBLE_DEVICES=1 


CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh /root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_vehicle.py  2

python ./tools/train.py /root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_visbilebody.py

python ./tools/train.py /root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_vehicle.py

-----------------------------------retinaface
python /root/data/gvision/detectron2-master/projects/Retinaface/train_net.py /root/data/gvision/detectron2-master/projects/Retinaface/configs/facetron/retinaface_r_50_3x.yaml


---------------------------------DetectoRS

train
python tools/train.py workdir/my_DetectoRS.py \
	--work_dir /root/data/gvision/DetectoRS-master/workdir/output/test 

test
python tools/test.py workdir/my_DetectoRS.py  \
	"/root/data/gvision/DetectoRS-master/workdir/rs_output/epoch_3.pth"\
  --out "/root/data/gvision/DetectoRS-master/workdir/output/my_p/results.pkl"
./tools/dist_test.sh configs/DetectoRS/DetectoRS_mstrain_split_person_ms.py \
		"/root/data/gvision/dataset/rs_output/epoch_3.pth"\
		3
---------------------------------CrowdDet
python train.py -md /root/data/gvision/CrowdDet-master/model/rcnn_fpn_baseline -r 
python test.py -md rcnn_fpn_baseline -r /root/data/gvision/CrowdDet-master/pretrained/rcnn_fpn_baseline_mge.pth
python test.py -md rcnn_emd_refine -r /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/model_dump/dump-29.pth


python inference.py -md rcnn_emd_refine -r rcnn_emd_refine -i /root/data/gvision/CrowdDet-master/data/panda/14_01/image_test/14_OCT_Habour_IMG_14_01___0.5__2816__3072.jpg
python my_inference.py -md rcnn_emd_refine -r rcnn_emd_refine -i /root/data/gvision/dataset/predict/person/test_person.json
python my_inference_48.py -md rcnn_emd_refine -r 48 -i /root/data/gvision/dataset/predict/person/test_person.json

python train.py --model_dir /root/data/gvision/CrowdDet-master/model/rcnn_fpn_baseline --resume_weights /root/data/gvision/CrowdDet-master/pretrained/rcnn_emd_refine_mge.pth
python visulize_json.py \
	-f /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/eval_dump/dump-rcnn_emd_refine_mge.pth.json \
	-n 3 \
	-s /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/visual



python eval_json.py -f /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/eval_dump/dump-rcnn_emd_refine_mge.pth.json 
---------------------------------DETR
python projects/DETR/train_net.py --num-gpus 2 --config-file projects/DETR/configs/detr.res50.coco.multiscale.150e.yaml


data = dict(
    imgs_per_gpu=2*5,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=[
            "/root/data/gvision/dataset/train/train_center/image_annos/split_p.json",
        ],
        img_prefix='/root/data/gvision/dataset/train/train_center/image_train',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/root/data/gvision/dataset/predict/s0.5_t0.9_14/image_annos/coco_results_hw.json',
        img_prefix='/root/data/gvision/dataset/predict/s0.5_t0.9_14/image_test',
        pipeline=test_pipeline))

	
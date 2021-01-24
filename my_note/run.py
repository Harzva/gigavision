/root/data/rubzz/ruby/ruby_output3/split_train_person_panda_fafaxue_3category/img
/root/data/rubzz/ruby/ruby_output3/split_train_person_panda_fafaxue_3category/split_train_person_panda_fafaxue_3category.json


CUDA_VISIBLE_DEVICES=0,1,2 ./train_net.py --num-gpus 3 \
  --config-file /root/data/gvision/detectron2-master/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml --eval-only 

  CUDA_VISIBLE_DEVICES=0,1,2 ./train_net.py --num-gpus 3 \
  --config-file /root/data/gvision/detectron2-master/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml \
  --resume /root/data/gvision/detectron2-master/workdir/output/my_end_ms_panda/model_0009999.pth
-------------------------------------------------*************************Pedestron
python tools/demo.py /root/data/gvision/Pedestron/configs/elephant/crowdhuman/cascade_hrnet.py \
		/root/data/gvision/Pedestron/workdir/pretrained/epoch_19_crowdhuman2.pth.stu  \
			 --input_img_dir /root/data/rubzz/ruby/ruby_output/test/person/split_test_method2_bigimageto1536_2 \
			--output_dir  workdir/output/big_visual2/

python tools/my_demo.py /root/data/gvision/Pedestron/configs/elephant/eurocity/cascade_hrnet.py \
		/root/data/gvision/Pedestron/workdir/pretrained/EuroCity_Persons_epoch_147.pth.stu  \
			 --input_img_dir /root/data/rubzz/ruby/ruby_output/test/person/split_test_method2_bigimageto1536_2 \
			--output_dir  workdir/output/eurocity/


python tools/my_demo.py /root/data/gvision/Pedestron/configs/elephant/cityperson/mgan_vgg.py \
		/root/data/gvision/Pedestron/workdir/pretrained/epoch_1.pth  \
			 --input_img_dir /root/data/rubzz/ruby/ruby_output/test/person/split_test_method2_bigimageto1536_2 \
			--output_dir  workdir/output/cityperson/



python tools/my_demo.py /root/data/gvision/Pedestron/configs/elephant/crowdhuman/cascade_hrnet.py \
		/root/data/gvision/Pedestron/workdir/pretrained/epoch_19.pth.stu  \
			 --input_img_dir /root/data/gvision/dataset/crop/test1 \
			--output_dir  workdir/output/crop/test1


python tools/demo.py /root/data/gvision/Pedestron/configs/elephant/cityperson/ga_retinanet_ResNeXt101.py \
		/root/data/gvision/Pedestron/workdir/pretrained/CityPerson_RetinaNet_with_Guided_Anchoring.stu  \
			 --input_img_dir /root/data/rubzz/ruby/ruby_output/test/person/split_test_method2_bigimageto1536_2 \
			--output_dir  workdir/output/cityperson/

python ./tools/test_crowdhuman.py /root/data/gvision/Pedestron/configs/elephant/crowdhuman/cascade_hrnet.py /root/data/gvision/Pedestron/workdir/pretrained/epoch_19.pth.stu \
 --out result_humanrowd.json 

python ./tools/test_crowdhuman.py configs/elephant/crowdhuman/cascade_hrnet.py ./pretrained/epoch_ 19 20\
 --out result_crowdhuman_meanteachers.json --mean_teacher 
				
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29505 ./tools/dist_test.sh configs/elephant/crowdhuman/cascade_hrnet.py /home/ubuntu/Documents/hzh/gigavision/Pedestron/epoch_19.pth.stu 4 \
	--out /home/ubuntu/Documents/hzh/gigavision/Pedestron/work_dirs/crowdhuman_cascade_rcnn_hrnet/result_humanrowd.pkl

CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh /root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco.py   2
python tools/train.py /root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco.py 

/root/data/gvision/mmdetection-master/workdir/detectors_1_2_fafaxue/output/detectors_ep30_12_fafaxue.bbox.json
---------------------------------**********************************mmdetection2		

python tools/test.py /root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_all.py \
    /root/data/gvision/mmdetection-master/workdir/detectors_1_2_fafaxue/latest.pth \
 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_1_2_fafaxue/output/detectors_ep30_12_fafaxue" \
    --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_1_2_fafaxue/visual	

CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/detectors_123_fafaxue_resume/detectors_htc_r50_1x_coco_123_resume.py \
	/root/data/gvision/mmdetection-master/workdir/detectors_123_fafaxue_resume/latest.pth \
    3 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_123_fafaxue_resume/output/detectors_123resume_fafaxue" \
    --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_123_fafaxue_resume//visual/

CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_fullbody.py \
	/root/data/gvision/mmdetection-master/workdir/pretrained/detectors_epoch_30.pth \
    3 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_fullbody_method2/output/detectors_epoch_30_fafaxue" \
    --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_fullbody_method2/visual/fafaxue 		



CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_fullbody.py \
	/root/data/gvision/mmdetection-master/workdir/pretrained/detectors_epoch_30.pth \
    3 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_fullbody_method2/output/detectors_epoch_30_fafaxue" \
    --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_fullbody_method2/visual/fafaxue 																																																																																																							
	CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_fullbody.py \
	/root/data/gvision/mmdetection-master/workdir/pretrained/detectors_epoch_30.pth \
    3 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_fullbody_method2/output/detectors_epoch_30_unsure" \
    --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_fullbody_method2/visual/fafaxue 

    if test:
        outpath="/root/data/gvision/my_merge/fusion_results"
    else:
        outpath="/root/data/gvision/final_merge/fusion_results"
	 CUDA_VISIBLE_DEVICES=0 PORT=29503 ./tools/dist_test.sh /root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_without17.py \
		/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/coco_bicycle_and_panda_else.pth \
	    1 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/output/ecoco_bicycle_and_panda_else/split_reuslt_detectros_else_without17" 
	
CUDA_VISIBLE_DEVICES=1 PORT=29500 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_17.py \
		/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/coco_bicycle_and_panda_else.pth \
    1 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/output/coco_bicycle_and_panda_else/split_reuslt_detectros_else_17" 


CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_else.py \
		/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/coco_bicycle_and_panda_else.pth \
    3 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/output/coco_bicycle_and_panda_else/split_reuslt_detectros_else" 

CUDA_VISIBLE_DEVICES=1 PORT=29501 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_else.py\
		/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/coco_bicycle_and_panda_else.pth \
    1 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/output/coco_bicycle_and_panda_else/split_reuslt_detectros_else" 


CUDA_VISIBLE_DEVICES=1,2  PORT=29501 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_cascade_rcnn_r50_1x_coco_3.py \
		/root/data/gvision/mmdetection-master/workdir/detectors_3_init/latest.pth \
   2 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_3_crowdhuman_init/cascade_3_crowdhuman" \
   --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_3_crowdhuman_init/visual 


CUDA_VISIBLE_DEVICES=1,2  PORT=29501 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_cascade_rcnn_r50_1x_coco_3_crowdhuman_resume.py \
		/root/data/gvision/mmdetection-master/workdir/detectors_3_crowdhuman_method2_resume/latest.pth \
   2 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_3_crowdhuman_method2_resume/cascade_3_crowdhuman_resume_3" \
   --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_3_crowdhuman_method2_resume/visual 

CUDA_VISIBLE_DEVICES=1,2  PORT=29502 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_train_vehicle_tourist.py \
		/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2_tour/epoch_12.pth \
	2 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2_tour/tour" \
 --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2_tour/visual 


CUDA_VISIBLE_DEVICES=0,1,2  PORT=29503 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/detectors_1_3_fafaxue_data/detectors_cascade_rcnn_r50_1x_coco_3_13.py \
		/root/data/gvision/mmdetection-master/workdir/detectors_1_3_fafaxue_data/latest.pth \
	3 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_1_3_fafaxue_data/test" \
 --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_1_3_fafaxue_data/visual






CUDA_VISIBLE_DEVICES=0,1,2  PORT=29503 ./tools/dist_test.sh 	/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2_elema/detectors_htc_r50_1x_coco_train_vehicle_elema.py \
		/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2_elema/latest.pth \
	3 --format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2_elema/output/test" \
 --show-dir /root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2_elema/visual


CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_train.sh /root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_all.py 3 --no-validate
CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_train.sh  /root/data/gvision/mmdetection-master/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py 3 --no-validate
CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_train.sh  /root/data/gvision/mmdetection-master/workdir/config/detectors_cascade_rcnn_r50_1x_coco_3_crowdhuman_resume.py 3 --no-validate


CUDA_VISIBLE_DEVICES=0,1,2 PORT=29501 ./tools/dist_train.sh  /root/data/gvision/mmdetection-master/workdir/config/detectors_htc_r50_1x_coco_train_vehicle_else.py 3 --no-validate

CUDA_VISIBLE_DEVICES=1,2  PORT=29502 ./tools/dist_train.sh  /root/data/gvision/mmdetection-master/workdir/config/detectors_cascade_rcnn_r50_1x_coco_elema.py 2 --no-validate



	
		3 \
	--format-only --options "jsonfile_prefix=/root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/output/split_reuslt_detectros_big_nms" 
	--show-dir /root/data/gvision/mmdetection-master/workdir/detectors_vehicle_method2/visual
--out 'split_reuslt_detectros_method2_nms.pkl' \
detectron2
CUDA_VISIBLE_DEVICES=0,1,2 python /root/data/gvision/detectron2-master/projects/Retinaface/train_net.py --num-gpus 3 --resume /root/data/gvision/detectron2-master/workdir/output/my_head_retinaface_ms_panda/model_0009347.pth

10.170.23.49
实际任务一定不是按这样划分的呀，不是应该每个场景都给一些带有标签信息的图片，然后18个场景进行预测吗，固定的摄像头就会有固定的背景信息，实际生活种
同样是标数据，最好的方法就是每个场景都标几张，然后训练，最后预测这些固定的场景。
-----------------------------pyramidbox
python /root/data/gvision/head_model/2018--ZJUAI--PyramidBoxDetector-master/test_yuncong_our_test2.py \
--resume '/root/data/gvision/head_model/2018--ZJUAI--PyramidBoxDetector-master/weights/best_mod_Res50_pyramid.pth'
python /root/data/gvision/head_model/2018--ZJUAI--PyramidBoxDetector-master/test_yuncong_our_test2.py \
--resume '/root/data/gvision/head_model/2018--ZJUAI--PyramidBoxDetector-master/weights/best_our_Res50_pyramid_aug.pth'

-------------------------------visuasl
 python visualize.py json/20200707_005329.log.json 
python tools/voc_eval_visualize.py result.pkl ./configs/faster_rcnn_r101_fpn_1x.py
-----------------------------------D2det

 CUDA_VISIBLE_DEVICES=0,1,2 ./tools/dist_test.sh /root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_vehicle.py     /root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_vehicle/latest.pth     3 --format_only --options "jsonfile_prefix=/root/data/gvision/D2Det-master/workdir/otput/D2Det_detection_r101_fpn_dcn_2x_vehicle/outputs_results_dyybigm2"
--options test_cfg.rcnn.score_thr=0.5    test_cfg.rcnn.nms.iou_thr=0.3 test_cfg.rcnn.max_per_img=50
/opt/conda/bin/python /root/data/gvision/D2Det-master/demo/demovisible.py --score_thr 0.5 --max_per_img 50 --iou_thr 0.3
/opt/conda/bin/python /root/data/gvision/D2Det-master/demo/demovehcile.py --score_thr 0.01 --max_per_img 50 --iou_thr 0.99

python tools/test.py /root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_visbilebody_test_noms.py \
/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/net_output/epoch_8.pth \
--out /root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_noms.pkl \
--format_only \
--options jsonfile_prefix="/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_noms" 

python tools/test.py /root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_visbilebody.py \
/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/net_output/epoch_8.pth \
--out /root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_noms.pkl \
--format_only \
--options jsonfile_prefix="/root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_visiblebody/results_output/det_results_dyy" 

CUDA_VISIBLE_DEVICES=1 ./tools/dist_test.sh \
	/root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_visbilebody.py  \
    /root/data/gvision/D2Det-master/workdir/otput/D2Det_detection_r101_fpn_dcn_2x_visiblebody/epoch_8.pth \
    1 --format_only --options "jsonfile_prefix=/root/data/gvision/D2Det-master/workdir/otput/D2Det_detection_r101_fpn_dcn_2x_visiblebody/outputs_results_1401"

CUDA_VISIBLE_DEVICES=1 ./tools/dist_test.sh \
python ./tools/test.py \
CUDA_VISIBLE_DEVICES=0,1,2 ./tools/dist_test.sh \
	/root/data/gvision/D2Det-master/configs/D2Det/my_D2Det_detection_r101_fpn_dcn_2x_vehicle.py \
    /root/data/gvision/D2Det-master/workdir/output/D2Det_detection_r101_fpn_dcn_2x_vehicle/latest.pth \
    3 --format_only --options "jsonfile_prefix=/root/data/gvision/D2Det-master/workdir/otput/D2Det_detection_r101_fpn_dcn_2x_vehicle/outputs_results_dyy"

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
------------------------------------------------CrowdDet
CUDA_VISIBLE_DEVICES=0,1 python train.py -md rcnn_emd_refinetp
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -md rcnn_emd_refinet
CUDA_VISIBLE_DEVICES=0,1 python train.py -md rcnn_emd_refine -r /root/data/gvision/CrowdDet-master/pretrained/rcnn_emd_refine_mge.pth
python test.py -md rcnn_fpn_baseline -r /root/data/gvision/CrowdDet-master/pretrained/rcnn_fpn_baseline_mge.pth
python my_test.py -md rcnn_emd_refine -r /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/model_dump/dump-29.pth 
python test.py -md rcnn_emd_refinet -r /root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/model_dump/dump-vbox_train30.pth -s vboxdump_30 -d 0-1-2 
python test.py -md rcnn_emd_refine -r /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/model_dump/dump-29.pth -s testalldump-29.pth -d 0-1-2 
python test.py -md rcnn_emd_refine -r /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/model_dump/dump-49.pth -s testalldump-49.pth -d 0-1-2 

python test.py -md rcnn_emd_refinet -r /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/model_dump/dump-49.pth -s testalldump-49.pth -d 0-1-2 

python test.py -md rcnn_emd_refinet -r /root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/model_dump/dump-vbox_train30.pth -d 0-1-2  -s bigvbox -n 30

python inference.py -md rcnn_emd_refine -r rcnn_emd_refine -i /root/data/gvision/CrowdDet-master/data/panda/14_01/image_test/14_OCT_Habour_IMG_14_01___0.5__2816__3072.jpg
python my_inference.py -md rcnn_emd_refine -r rcnn_emd_refine -i /root/data/gvision/dataset/predict/person/test_person.json
python my_inference_29.py -md rcnn_emd_refine -r 29 -i /root/data/gvision/dataset/predict/person/test_person.json

python train.py --model_dir /root/data/gvision/CrowdDet-master/model/rcnn_fpn_baseline --resume_weights /root/data/gvision/CrowdDet-master/pretrained/rcnn_emd_refine_mge.pth
python visulize_json.py \
	-f /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/eval_dump/dump-rcnn_emd_refine_mge.pth.json \
	-n 3 \
	-s /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/visual



python eval_json.py -f /root/data/gvision/CrowdDet-master/model/rcnn_emd_refine/outputs/eval_dump/dump-rcnn_emd_refine_mge.pth.json 
-----------------------------------------------DETR
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

	
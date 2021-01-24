${CONFIG_FILE}：代表着 config/里面文件的位置，比如configs/mask_rcnn_r50_fpn_1x.py。
${CHECKPOINT_FILE}：代表着模型权重所在的位置。比如checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth。
[--out ${RESULT_FILE}]：代表着测试生成的文件的位置。
[--eval ${EVAL_METRICS}]：所选用的测试方式，EVAL_METRICS 下面有解析。
${GPU_NUM}：GPU数量，比如 --gpu 2代表用两个GPU进行训练或者测试。
python tools/train.py configs/DetectoRS/DetectoRS_mstrain_split_person_ms.py 
python tools/test.py configs/DetectoRS/DetectoRS_mstrain_split_person_ms.py  \
	"/root/data/gvision/dataset/rs_output/epoch_3.pth"\
  --out "/root/data/gvision/dataset/rs_output/my_p/results.pkl"
./tools/dist_test.sh configs/DetectoRS/DetectoRS_mstrain_split_person_ms.py \
		"/root/data/gvision/dataset/rs_output/epoch_3.pth"\
		3
python /root/data/gvision/detectron2-master/projects/TridentNet/train_net.py \
	"/root/data/gvision/detectron2-master/projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml"\
	"/root/data/gvision/detectron2-master/projects/TridentNet/output"
python /root/data/gvision/detectron2-master/projects/TridentNet/train_net.py  --config-file "/root/data/gvision/detectron2-master/projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml" --eval-only --num-gpus 1
python /root/data/gvision/detectron2-master/projects/TridentNet/train_net.py  --config-file "/root/data/gvision/detectron2-master/projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml" --eval-only --num-gpus 1

./tools/dist_train.sh configs/faster_rcnn/my_faster_rcnn_config.py 2
  --jsonout "/root/data/gvision/dataset/mm_output/my_pv_iou/my_predict/results.json"\
python tools/test.py configs/faster_rcnn/my_faster_rcnn_config.py \
  "/root/data/gvision/dataset/mm_output/my_pv_iou/epoch_1.pth" \
  --out "/root/data/gvision/dataset/mm_output/my_pv_iou/results.pkl"\
  --show \
  --no-validate\
  --show-dir "/root/data/gvision/dataset/mm_output/my_pv_iou/my_predict_split_visual"
python tools/test_robustness.py configs/faster_rcnn/my_faster_rcnn_config.py \
  "/root/data/gvision/dataset/mm_output/my_pv_iou/epoch_1.pth" \
  --out "/root/data/gvision/dataset/mm_output/my_pv_iou/results.pkl"\
  --show \
  --no-validate\
  --show-dir "/root/data/gvision/dataset/mm_output/my_pv_iou/my_predict_split_visual"

python tools/test.py /xxxx/xxxx/mmdetection/configs/guided_anchoring/ga_faster_x101_32x4d_fpn_1x.py /xxxx/xxxx/mmdetection/work_dirs3/epoch_1.pth --json_out res091420 --eval bbox



# single-gpu testing
python tools/test_robustness.py configs/faster_rcnn/my_faster_rcnn_config.py \
  "/root/data/gvision/dataset/mm_output/my_pv_train/latest.pth" \
  --out "/root/data/gvision/dataset/mm_output/my_pv_train/results.pkl"\
  --jsonout "/root/data/gvision/dataset/mm_output/my_pv_faster/results.json"\
  --show \
  --eval \
  --show-dir "/root/data/gvision/dataset/mm_output/my_pv_faster/my_predict_split_visual"
  --launcher 'pytorch'


10.170.34.47
ssh -p 29250 root@192.168.137.15
431vpnstart
ssh -p 12345 root@192.168.29.136
sudo route add -net 192.168.29.0 netmask 255.255.255.0 gw 10
gpustat
watch --color -n1 gpustat -cpu
pipeline
- 数据读取
- 数据预处理

- 创建模型
- 训练模型

- 模型评估
- 模型调参
ETD是Estimated Time of Departure的首字母简称,意思是预计离港时间、预计离开时间。 ETA是Estimated Time of Arrival的首字母简称,意思为预计到港时间elapsed time_实耗时间
baseline 
conda env export > your_env.yaml
conda env export -- name 
conda env create -f name.yml

conda info --env
conda create -n name --clone path
python setup.py install
python setup.py develop
pip download -d \home\ubuntu\tesla -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
	$ pip download -d DIR -r requirements.txt
$ pip wheel -w DIR -r requirements.txt
 pip3 download 
 	--no-deps
    --platform linux_x86_64
    --python-version 36
    --implementation cp  
    --abi cp36m    
     -r requirements.txt -d pk
		
在这里可以看到，首先对输入图像提取特征，然后将特征送到rpn_head,之后计算rpn_loss然后通过get_bboxes()取到proposal_list。
先获取anchors然后通过get_bboxes_single来获取候选框，主要有两个比较重要的操作，按照score排序取最大的k个boxes，取到topk个bboxes过后通过delta2bbox()转换成(x1,y1,x2,y2)的格式。

现在按照one-stage和two-stage的分法已经有些过时了，毕竟two-stage可以加Cascade，one-stage可以加Refine；还有一堆Anchor-base和Anchor-free的方法。想要在目标检测任务中定位更准确，当然模型越大算法越新，越有优势，但是实际中要看数据集，要根据数据集中目标尺度分布进行调参，以及选择合适的数据增强方式。
__all__ = [
    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
    'DETECTORS', 'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss', 'build_detector'
]

__all__backones = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net'
]
__all__dense_heads = [
    'AnchorHead', 'GuidedAnchorHead', 'FeatureAdaption', 'RPNHead',
    'GARPNHead', 'RetinaHead', 'RetinaSepBNHead', 'GARetinaHead', 'SSDHead',
    'FCOSHead', 'RepPointsHead', 'FoveaHead', 'FreeAnchorRetinaHead',
    'ATSSHead', 'FSAFHead', 'NASFCOSHead', 'PISARetinaHead', 'PISASSDHead'
]
__all__ detectors= [
    'ATSS', 'BaseDetector', 'SingleStageDetector', 'TwoStageDetector', 'RPN',
    'FastRCNN', 'FasterRCNN', 'MaskRCNN', 'CascadeRCNN', 'HybridTaskCascade',
    'RetinaNet', 'FCOS', 'GridRCNN', 'MaskScoringRCNN', 'RepPointsDetector',
    'FOVEA', 'FSAF', 'NASFCOS'
]

__all__losses = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'sigmoid_focal_loss',
    'FocalLoss', 'smooth_l1_loss', 'SmoothL1Loss', 'balanced_l1_loss',
    'BalancedL1Loss', 'mse_loss', 'MSELoss', 'iou_loss', 'bounded_iou_loss',
    'IoULoss', 'BoundedIoULoss', 'GIoULoss', 'GHMC', 'GHMR', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'L1Loss', 'l1_loss', 'isr_p',
    'carl_loss'
]

__all__necks = [
    'FPN', 'BFP', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN', 'NASFCOS_FPN'
]


__all__ roi_head= [
    'BaseRoIHead', 'CascadeRoIHead', 'DoubleHeadRoIHead', 'MaskScoringRoIHead',
    'HybridTaskCascadeRoIHead', 'GridRoIHead', 'ResLayer', 'BBoxHead',
    'ConvFCBBoxHead', 'Shared2FCBBoxHead', 'Shared4Conv1FCBBoxHead',
    'DoubleConvFCBBoxHead', 'FCNMaskHead', 'HTCMaskHead', 'FusedSemanticHead',
    'GridHead', 'MaskIoUHead', 'SingleRoIExtractor', 'PISARoIHead'
]


backbones = Registry('backbone')
@backbones.register_module()
class ResNet:
     pass

backbones = Registry('backbone')
@backbones.register_module(name='mnet')
class MobileNet:
     pass

backbones = Registry('backbone')
class ResNet:
     pass
backbones.register_module(ResNet)


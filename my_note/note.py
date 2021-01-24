/root/data/gvision/dataset/UAC/UAC_test/test/annotations.json


/root/data/gvision/detectron2-master/workdir/output/my_head_retinaface_ms_panda/my_predict/final_nms0.7_fs0.1_all.json

test_json= "/root/data/rubzz/ruby/ruby_output2/test_person_unsure_cell.json"
test_image_path="/root/data/rubzz/ruby/ruby_output2/test_person_unsure_cell"
{
  "categories": [
    {
      "supercategory": "none",
      "id": 1,
      "name": "small car"
    },
    {
      "supercategory": "none",
      "id": 2,
      "name": "midsize car"
    },
   {
      "supercategory": "none",
      "id": 3,
      "name": "large car"
    },
    {
      "supercategory": "none",
      "id": 4,
      "name": "bicycle"
    },
    {
      "supercategory": "none",
      "id": 5,
      "name": "baby carriage"
    },
    {
      "supercategory": "none",
      "id": 6,
      "name": "motorcycle"
    },
    {
      "supercategory": "none",
      "id": 7,
      "name": "tricycle"
    },
    {
      "supercategory": "none",
      "id": 8,
      "name": "electric car"
    }
  ],　
	[[3552,0.5,100],[4652,0.25,200],[7052,0.1,400]],      # 14   ,[5904,0,0]
    [[2004,0.8,300],[2404,0.5,350],[3304,0.4,400],[4504,0.25,500],[6704,0.16,700],[10204,0.13,800],[14457,0.08,1200]],      # 15
    [[0,1,500],[671,0.2,550],[2229,0.2,600],[5209,0.2,600],[8009,0.2,600],[10809,0.2,700],[13509,0.15,700],[17143,0.12,700]],# 16_1
    [[0,0.15,750],[2761,0.15,750],[6595,0.12,800],[11662,0.1,800],[18012,0.1,800]],      # 16_2
    [[5990,0.8,200],[6590,0.32,400],[8290,0.12,600]],      # 17
    [[4257,1,100],[4757,0.25,300],[6957,0.1,600]],      # 18_1
    [[3813,0.18,400],[6957,0.1,600]],      # 18_2
import time
import sys
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf8')
 /root/data/rubzz/ruby/fafaxueyun_output/train_vhicle_smallcar/train_bbox_vehicle_else_1400_4567.json
	/root/data/rubzz/ruby/fafaxueyun_output/train_vhicle_smallcar/img
class Meiju100Pipeline(object):
    def process_item(self, item, spider):
        today = time.strftime('%Y%m%d',time.localtime())
        fileName = today + 'movie.txt'
        with open(fileName,'a') as fp:
            fp.write(item['storyName'][0].encode("utf8") + '\t' + item['storyState'][0].encode("utf8") + '\t' + item['tvStation'][0] + '\t' + item['updateTime'][0] + '\n')
        return item
	asdfg
ssh -p 19452 root@192.168.137.13
ssh -p 12313 root@192.168.137.12
tensorboard --logdir 
scp -r split* root@192.168.137.12
scp  -r -P 25738 split* root@192.168.137.12:/root/data/
scp  -r  -P 19452 root@192.168.137.13:/root/data/
	
	ping https://github.com/saic-vul/iterdet/releases/download/v2.0.0/crowd_human_full_faster_rcnn_r50_fpn_2x.pth
	
52.217.44.68 github-production-release-asset-2e65be.s3.amazonaws.com
219.76.4.4 s3.amazonaws.com
219.76.4.4 github-cloud.s3.amazonaws.com
	
Host sys
    HostName 10.170.60.203
    User 
scp *  ubuntu@10.170.60.203:/home/ubuntu/Documents/hzh/gigavision/
ssh ubuntu@10.170.60.203
 scp /root/data/rubzz/ruby/ ubuntu@10.170.62.117:/home/ubuntu/crowdhuman/
	
	scp det_submission.txt ubuntu@10.170.62.117:/home/ubuntu/crowdhuman/
ssh ubuntu@action101.ddns.net
ssh ubuntu@10.170.60.203

	/root/data/gvision/dataset/raw_data/image_train
	/root/data/gvision/dataset/raw_data/image_annos
10.170.65.227
source /root/data/crowd/bin/activate

vncserver -kill :1

fuser -v /dev/nvidia*

vncserver -geometry 1920x1080 :3
service vncserver start
import torch
print(torch.cuda.is_available())
from torch.backends import cudnn
print(cudnn.is_available()) 
10.170.25.53
[Desktop Entry]
Name=vscode
Comment=firefox
Exec=/opt/VSCode-linux-x64/code
Icon=/opt/VSCode-linux-x64/resources/app/resources/linux/code.png
Terminal=false
Type=Application
Categories=Application;
Encoding=UTF-8
StartupNotify=true
	
	
	{"height":, "ID":,"width":,"dtboxes":[{"score":,"tag":,"box":[]}]}

conda create -p /root/data/conda/envs/mm2 --clone base
conda create -p mm --clone /root/data/conda/envs/g1.4 
conda create -p /root/data/conda/envs/g1.4 -n base --clone opt/conda/envs/g1.4
conda create -n mm --clone base
conda create -p /root/data/conda/envs/g1.4 --clone opt/conda/
cp -r /opt/conda/
conda create  -p /opt/environment/.conda/envs/env_name  python=2.7
source /opt/conda/bin/activate
conda activate pt15
conda install cudnn=7.6.5 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/
conda install cudatoolkit=10.1 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/
[i for i in a if i["name"]=='motorcycle'][0]["id"]
.condarc
  active environment : base
    active env location : /opt/conda
            shell level : 1
       user config file : /root/.condarc
 populated config files : /root/.condarc
          conda version : 4.7.11
    conda-build version : 3.17.8
         python version : 3.6.8.final.0
       virtual packages : __cuda=10.2
       base environment : /opt/conda  (writable)
           channel URLs : https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/noarch
                          https://repo.anaconda.com/pkgs/main/linux-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/linux-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /opt/conda/pkgs
                          /root/.conda/pkgs
       envs directories : /opt/conda/envs
                          /root/.conda/envs
               platform : linux-64
             user-agent : conda/4.7.11 requests/2.21.0 CPython/3.6.8 Linux/4.4.0-142-generic ubuntu/16.04.6 glibc/2.23
                UID:GID : 0:0
             netrc file : None
           offline mode : False
			
			
		     active environment : None
            shell level : 0
       user config file : /root/.condarc
 populated config files : /root/.condarc
          conda version : 4.7.11
    conda-build version : 3.17.8
         python version : 3.6.8.final.0
       virtual packages : __cuda=10.2
       base environment : /opt/conda  (writable)
           channel URLs : https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/noarch
                          https://repo.anaconda.com/pkgs/main/linux-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/linux-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /root/data/g1.4
       envs directories : /root/data/g1.4
                          /opt/conda/envs
                          /root/.conda/envs
               platform : linux-64
             user-agent : conda/4.7.11 requests/2.21.0 CPython/3.6.8 Linux/4.4.0-142-generic ubuntu/16.04.6 glibc/2.23
                UID:GID : 0:0
             netrc file : None
           offline mode : False
	
			



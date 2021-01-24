import os.path as osp
import json
from glob import glob
import mmcv
from PIL import Image
from tqdm import tqdm


def cvt_annotations(json_bbox, out_file):
    images = []
    annotations = []
    anno_id = 1
    for img_path in json_bbox:
        image_dict = json_bbox[img_path]
        img_name = img_path
        h = image_dict['image size']['height']
        w = image_dict['image size']['width']
        img_id = image_dict['image id']
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': img_id,
                   'bbox': bbox, 'category_id': 1, 'id': anno_id, 'ignore': 0}
            anno_id += 1
            annotations.append(ann)

    categories = [{"name": 'vehicle visible part', "id": 1}]
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    json_path = '/root/panda/annotations/vehicle_bbox_test.json'
    with open(json_path, "r") as fp:
        json_data = json.load(fp)
    cvt_annotations(json_data, '/root/panda/annotations/vehicle_test.json')
    print('Done!')


if __name__ == '__main__':
    main()

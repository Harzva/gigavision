import os.path as osp
import json
from glob import glob
import mmcv
from PIL import Image
from tqdm import tqdm

label_ids = {'motorcycle': 1, 'small car': 1, 'bicycle': 1,
             'baby carriage': 1, 'midsize car': 1, 'large car': 1, 'electric car': 1,
             'tricycle': 1, 'fake': 0, 'ignore': 0, 'vehicles': 0,
             'unsure': 0}


def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def get_annotation(image_dict, img_id, anno_id, width, height):
    annotation = []
    for obj in image_dict['objects list']:
        bnd_box = obj['rect']
        ignore = 0 if label_ids[obj['category']] else 1
        xmin = int(bnd_box['tl']['x'] * width)
        ymin = int(bnd_box['tl']['y'] * height)
        xmax = int(bnd_box['br']['x'] * width)
        ymax = int(bnd_box['br']['y'] * height)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w * h
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [xmin, ymin, w, h],
            "category_id": 1,
            "id": anno_id,
            "ignore": ignore})
        anno_id += 1
    return annotation, anno_id


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

        annos, anno_id = get_annotation(image_dict, img_id, anno_id, w, h)
        annotations.extend(annos)

    categories = [{"name": 'vehicle visible part', "id": 1}]
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    json_path = '/root/panda/annotations/vehicle_bbox_train.json'
    with open(json_path, "r") as fp:
        json_data = json.load(fp)
    cvt_annotations(json_data, '/root/panda/annotations/vehicle_train.json')
    print('Done!')


if __name__ == '__main__':
    main()

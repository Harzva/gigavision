import os.path as osp
import json
from glob import glob

import mmcv
from PIL import Image
from tqdm import tqdm


label_ids = {'person': 1, 'ignore': 1,
             'people': 1, 'crowd': 1, 'fake person': 1}


def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def get_annotation(image_dict, img_id, anno_id, width, height):
    annotation = []
    for obj in image_dict['objects list']:
        name = obj['category']
        # print(obj)
        bnd_box = obj['rects']['visible body'] if 'rects' in obj else obj['rect']
        xmin = int(bnd_box['tl']['x'] * width)
        ymin = int(bnd_box['tl']['y'] * height)
        xmax = int(bnd_box['br']['x'] * width)
        ymax = int(bnd_box['br']['y'] * height)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w * h
        segmentation = get_segmentation([xmin, ymin, w, h])
        ignore = 0 if name == 'person' else 1
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
    # 'person': 1
    categories = [{"name": 'person', "id": 1}]
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    json_path = '/data/panda/annotations/person_bbox_train.json'
    with open(json_path, "r") as fp:
        json_data = json.load(fp)
    cvt_annotations(json_data, '/data/panda/annotations/person_train_only.json')
    print('Done!')


if __name__ == '__main__':
    main()

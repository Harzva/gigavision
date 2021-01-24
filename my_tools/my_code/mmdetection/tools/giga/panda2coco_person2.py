import os.path as osp
import json
from glob import glob
import mmcv
from PIL import Image
from tqdm import tqdm

label_ids = {'head': 1, 'ignore': 1, 'person': 1,
             'people': 1, 'crowd': 1, 'fake person': 1,
             'visible body': 2, 'full body': 3}


def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def get_annotation(image_dict, img_id, anno_id, width, height):
    annotation = []
    for obj in image_dict['objects list']:
        if 'rects' in obj:
            for bnd_box in obj['rects']:
                category_id = label_ids[bnd_box]
                bnd_box = obj['rects'][bnd_box]
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
                    "category_id": category_id,
                    "id": anno_id,
                    "ignore": 0})
                anno_id += 1
        else:
            bnd_box = obj['rect']
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
                "ignore": 1})
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
    categories = [{"name": 'head', "id": 1}, {"name": 'visible body', "id": 2}, {"name": 'full body', "id": 3}]
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    json_path = '/root/panda/annotations/person_bbox_train.json'
    with open(json_path, "r") as fp:
        json_data = json.load(fp)
    cvt_annotations(json_data, '/root/panda/annotations/person_train.json')
    print('Done!')


if __name__ == '__main__':
    main()

import os.path as osp
import json
from glob import glob
import mmcv
from PIL import Image
from tqdm import tqdm

categories = [{"name": 'head', "id": 1}, {"name": 'visible body', "id": 2},
              {"name": 'full body', "id": 3}, {"name": 'vehicle visible part', "id": 4}]


def cvt_annotations(person_json_data, vehicle_json_data, out_file):
    images = []
    img_ids = set()
    for img in person_json_data['images']:
        images.append(img)
        img_ids.add(img['id'])
    for img in vehicle_json_data['images']:
        if img['id'] not in img_ids:
            img_ids.add(img['id'])
            images.append(img)

    anno_id = 1
    annotations = []
    for annotation in person_json_data['annotations']:
        annotation['id'] = anno_id
        anno_id += 1
        annotations.append(annotation)
    for annotation in vehicle_json_data['annotations']:
        annotation['id'] = anno_id
        annotation['category_id'] = 4
        anno_id += 1
        annotations.append(annotation)
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)


def main():
    person_path = '/root/panda/annotations/person_test.json'
    vehicle_path = '/root/panda/annotations/vehicle_test.json'
    with open(person_path, "r") as fp:
        person_json_data = json.load(fp)
    with open(vehicle_path, "r") as fp:
        vehicle_json_data = json.load(fp)
    cvt_annotations(person_json_data, vehicle_json_data, '/root/panda/annotations/all_test.json')
    print('Done!')


if __name__ == '__main__':
    main()

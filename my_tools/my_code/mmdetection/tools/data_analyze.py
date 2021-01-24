import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Analyze Json Log')
parser.add_argument(
    '--json_file',
    type=str,
    help='path of annotations json')
args = parser.parse_args()


def _main():
    with open(args.json_file) as fp:
        annotations = json.load(fp)
    images = annotations['images']
    bboxes = []
    for box in annotations['annotations']:
        img = images[box['image_id']]
        wh = [img['width'], img['height']]
        if img['height'] == 2160:
            for _ in range(2):
                bboxes.append(wh + box['bbox'])
        bboxes.append(wh + box['bbox'])
    bboxes = np.array(bboxes)
    bboxes = bboxes / bboxes[:, [1]] * 1000.0
    plt.hist(bboxes[:, 4], bins=100, label="")
    plt.show()
    plt.hist(bboxes[:, 5], bins=100, label="")
    plt.show()
    plt.hist(np.divide(bboxes[:, 4], bboxes[:, 5], where=bboxes[:, 5] != 0), bins=100, )
    plt.show()
    area = bboxes[:, 4] * bboxes[:, 5]
    plt.hist(area[area < 80000], bins=100)
    plt.show()
    print((area > 60000).sum())


if __name__ == "__main__":
    _main()

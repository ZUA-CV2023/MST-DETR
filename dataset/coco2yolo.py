import os
import json
from tqdm import tqdm
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='D:/Datasets/coco/annotations/instances_train2017.json', type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='D:/Datasets/coco-new/labels', type=str, help="specify where to save the output dir of labels")
args = parser.parse_args()

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)

def process_image(img, data, id_map, ana_txt_save_path):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    head, tail = os.path.splitext(filename)
    ana_txt_name = head + ".txt"
    f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
    for ann in data['annotations']:
        if ann['image_id'] == img_id:
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
    f_txt.close()

if __name__ == '__main__':
    json_file = args.json_path
    ana_txt_save_path = args.save_path

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    id_map = {}
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

    list_file = open(os.path.join(ana_txt_save_path, 'train2017.txt'), 'w')
    images = data['images']

    # 使用线程池处理每个图像
    with ThreadPoolExecutor(max_workers=10) as executor:
        for img in tqdm(images):
            executor.submit(process_image, img, data, id_map, ana_txt_save_path)
            list_file.write('D:/Datasets/coco/train2017/%s.jpg\n' % (os.path.splitext(img["file_name"])[0]))

    list_file.close()

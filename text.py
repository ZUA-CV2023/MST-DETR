# import os
#
# # COCO 数据集的类别名称
# coco_classes = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'
# ]
#
# # 标签文件夹路径
# label_dir = 'D:/Datasets/coco/labels'
# image_dir = 'D:/Datasets/coco/images'
#
# # 获取所有标签文件
# label_files = os.listdir(label_dir)
#
# for label_file in label_files:
#     # 读取每个标签文件
#     with open(os.path.join(label_dir, label_file), 'r') as f:
#         labels = f.readlines()
#
#     print(f"图片: {label_file.replace('.txt', '.jpg')} 的标签:")
#
#     # 解析标签文件
#     for label in labels:
#         label_info = label.strip().split()
#         class_id = int(label_info[0])  # YOLO 格式中的 class_id
#         class_name = coco_classes[class_id]  # 根据类别 ID 查找 COCO 类别名称
#
#         print(f"  类别 ID: {class_id}, 类别名称: {class_name}")
#
#     print()



coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

import os

# 标签文件夹路径
label_dir = 'path_to_txt_labels'

# COCO 数据集的类别数量
num_classes = len(coco_classes)

# 获取所有标签文件
label_files = os.listdir(label_dir)

# 遍历每个标签文件
for label_file in label_files:
    file_path = os.path.join(label_dir, label_file)

    # 读取每个标签文件
    with open(file_path, 'r') as f:
        labels = f.readlines()

    # 打印文件名
    print(f"检查文件: {label_file.replace('.txt', '.jpg')}")

    # 检查每行标签
    for line in labels:
        parts = line.strip().split()

        if len(parts) < 1:
            print(f"  错误: 标签行缺少数据 - {line.strip()}")
            continue

        try:
            class_id = int(parts[0])

            if class_id < 0 or class_id >= num_classes:
                print(f"  错误: 类别 ID 超出范围 - {class_id}")
            else:
                print(f"  正确: 类别 ID - {class_id}, 类别名称 - {coco_classes[class_id]}")
        except ValueError:
            print(f"  错误: 类别 ID 非整数 - {parts[0]}")
import os

def save_image_paths_by_category(folder_path, train_file, val_file,test_file):
    # 打开txt文件，准备写入
    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f, open(test_file, 'w') as test_f:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 判断文件是否是jpg格式的图片
            if filename.lower().endswith('.jpg'):
                # 获取文件的完整路径
                file_path = os.path.join(folder_path, filename)
                # 根据文件名的前缀判断分类
                if filename.lower().startswith('train'):
                    train_f.write(file_path + '\n')
                elif filename.lower().startswith('val'):
                    val_f.write(file_path + '\n')
                elif filename.lower().startswith('test'):
                    test_f.write(file_path + '\n')



if __name__ == '__main__':
    # 设置你要处理的文件夹路径和输出的txt文件路径
    folder_path = "D:/Datasets/UAV/images"
    train_file = "D:/Datasets/UAV/train.txt"
    val_file = "D:/Datasets/UAV/val.txt"
    test_file = "D:/Datasets/UAV/test.txt"
    save_image_paths_by_category(folder_path, train_file, val_file,test_file)

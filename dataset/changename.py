import os

def rename_images_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 判断文件是否是jpg格式的图片
        if filename.lower().endswith('.xml'):
            # 获取文件的完整路径
            old_file_path = os.path.join(folder_path, filename)
            # filename = old_file_path.split('_')[-1]
            # # 获取文件夹的名字
            # folder_name = os.path.basename(folder_path)
            # 构造新的文件名
            new_filename = f"{'train'}_{filename}"
            # 获取新的文件路径
            new_file_path = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")


if __name__ == "__main__":
    # 设置你要处理的文件夹路径
    folder_path = "D:/Datasets/UAV/train/xml"
    rename_images_in_folder(folder_path)

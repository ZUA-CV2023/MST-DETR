# # import os
# # from shutil import copy
# # from PIL import Image
# #
# # # 设置图片文件夹、文本文件夹和目标文件夹的路径
# # image_folder_path = 'D:/Datasets/coco/val2017'
# # text_folder_path = 'D:/Datasets/coco-new/labels'
# # destination_folder_path = 'D:/Datasets/coco_new/val_new'
# # sum = 0
# # # 确保目标文件夹存在
# # if not os.path.exists(destination_folder_path):
# #     os.makedirs(destination_folder_path)
# #
# # # 遍历图片文件夹中的所有文件
# # for image_file in os.listdir(image_folder_path):
# #     # 检查文件是否为图片
# #     if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
# #         # 提取不带扩展名的文件名
# #         image_name_without_extension = os.path.splitext(image_file)[0]
# #
# #         # 构建对应的文本文件名
# #         text_file = image_name_without_extension + '.txt'
# #
# #         # 检查文本文件是否存在
# #         if os.path.exists(os.path.join(text_folder_path, text_file)):
# #             sum = sum + 1
# #             # 构建完整的文本文件路径
# #             text_file_path = os.path.join(text_folder_path, text_file)
# #             # 构建目标路径
# #             destination_file_path = os.path.join(destination_folder_path, text_file)
# #
# #             # 复制文件到目标文件夹
# #             copy(text_file_path, destination_file_path)
# #             print(f"Copied '{text_file}' to '{destination_folder_path}'.")
# #             print("sum == ",sum)
# #         else:
# #             print(f"Text file '{text_file}' not found in '{text_folder_path}'.")
#
# import os
# from shutil import copy
#
# # 设置文件夹路径
# folder_a_path = 'D:/Datasets/coco/val'
# folder_b_path = 'D:/Datasets/coco_new/val_new'
#
# # 获取两个文件夹中所有.txt文件的列表
# txt_files_in_folder_a = {f for f in os.listdir(folder_a_path) if f.endswith('.txt')}
# txt_files_in_folder_b = {f for f in os.listdir(folder_b_path) if f.endswith('.txt')}
#
# # 找出在folder_a中但不在folder_b中的.txt文件
# txt_files_to_copy = txt_files_in_folder_a - txt_files_in_folder_b
# sum = 0
# # 遍历这些.txt文件并复制到folder_b
# for txt_file in txt_files_to_copy:
#     sum += 1
#     # 构建完整的文件路径
#     file_path = os.path.join(folder_a_path, txt_file)
#     destination_path = os.path.join(folder_b_path, txt_file)
#
#     # 复制文件
#     copy(file_path, destination_path)
#     print(f"Copied '{txt_file}' from '{folder_a_path}' to '{folder_b_path}'.")
#     print("sum = ",sum)
# # 如果没有.txt文件需要复制，则打印消息
# if not txt_files_to_copy:
#     print("No .txt files to copy. Both folders contain the same .txt files.")
#

import os
import numpy as np
import cv2
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
import xml.dom.minidom


def get_filename_without_extension(file_path):
    # 获取文件名（带后缀）
    file_name_with_extension = os.path.basename(file_path)
    # 去除文件后缀
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    return file_name_without_extension


def count_x_y_sub_images(subsize, overlap, H_img, W_img):
        # 计算 横x 纵y 方向上 需要提取多少个子图
        i, j = 1, 1 
        while(True):
            res_W = W_img - ((i*subsize) - ((i-1)*overlap))
            if res_W <= subsize - overlap:
                break
            else:
                i += 1
        while(True):
            res_H = H_img - ((j*subsize) - ((j-1)*overlap))
            if res_H <= subsize - overlap:
                break
            else:
                j += 1
        return i+1, j+1  # x方向个数, y方向个数


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    names = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        names.append(name)

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # 转换为左下角、右下角、右上角、左上角的顺序
        box = [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
        boxes.append(box)

    return boxes, names


def create_voc_xml(file_name, boxes, names):
    root = ET.Element("annotation")

    for box, name in zip(boxes, names):
        object_elem = ET.SubElement(root, "object")
        
        name_elem = ET.SubElement(object_elem, "name")
        name_elem.text = name

        bndbox_elem = ET.SubElement(object_elem, "bndbox")
        xmin_elem = ET.SubElement(bndbox_elem, "xmin")
        xmin_elem.text = str(box[0])
        ymin_elem = ET.SubElement(bndbox_elem, "ymin")
        ymin_elem.text = str(box[1])
        xmax_elem = ET.SubElement(bndbox_elem, "xmax")
        xmax_elem.text = str(box[2])
        ymax_elem = ET.SubElement(bndbox_elem, "ymax")
        ymax_elem.text = str(box[3])

    # 将 XML 格式化
    xml_str = ET.tostring(root, encoding='utf-8')
    xml_str = xml.dom.minidom.parseString(xml_str).toprettyxml()

    # 保存格式化后的 XML 文件
    with open(file_name, 'w') as f:
        f.write(xml_str)


def main(ImagePath, AnnotationPath, SavePath, subsize=1024, overlap=200, IoU_thresh=0.5, keep_no_object_images=False):
    # 获取图像和标签名称
    images_list = os.listdir(ImagePath)
    labels_list = os.listdir(AnnotationPath)
    images_list.sort()
    labels_list.sort()

    # 保存路径
    os.makedirs(SavePath, exist_ok=True)
    images_save_path = os.path.join(SavePath, "images")
    labels_save_path = os.path.join(SavePath, "labels")
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(labels_save_path, exist_ok=True)

    # 遍历所有图像
    for img_idx, image in enumerate(images_list):
        image = os.path.join(ImagePath, image)
        label = os.path.join(AnnotationPath, labels_list[img_idx])

        # 获取文件名
        file_name = get_filename_without_extension(image)

        # 读取标注
        boxes, names = parse_xml(label)

        # 读取图像
        img = cv2.imread(image)
        H_img = np.shape(img)[0]
        W_img = np.shape(img)[1]
        # print("Image Height:", H_img)
        # print("Image Width:", W_img)
        # 检测subsize的合法性
        if (subsize >= H_img) or (subsize >= W_img):
            print("Wrong subsize, subsize must < Height/Weight of the Image!!!")

        # 检测overlap合法性
        if overlap <= 0:
            print("Wrong overlap, overlap must > 0 !!!")
        if overlap >= subsize:
            print("Wrong overlap, overlap < subsize !!!")

        # 计算横x(W) 纵y(H) 方向上子图个数
        n_x, n_y = count_x_y_sub_images(subsize, overlap, H_img, W_img)

        # 遍历整个大图
        for i in range(n_x):  # x是W
            for j in range(n_y):  # y是H
                # 根据滑动窗口的位置 确定当前子图左上角的坐标
                if (i != n_x-1) and (j != n_y-1):
                    left_up = [i * (subsize - overlap), j * (subsize - overlap)]  # left_up=(x左上, y左上)
                elif (i == n_x-1) and ( j != n_y-1):
                    left_up = [W_img-subsize, j * (subsize - overlap)]
                elif (i != n_x-1) and ( j == n_y-1):
                    left_up = [i * (subsize - overlap), H_img-subsize]
                else:
                    left_up = [W_img-subsize, H_img-subsize]
                
                # 保存sub_img
                sub_img = img[left_up[1]:left_up[1]+subsize, left_up[0]:left_up[0]+subsize, :]
                cv2.imwrite(os.path.join(images_save_path, "{}_{}_{}.jpg".format(file_name, left_up[0], left_up[1])), sub_img)
                
                # 计算该sub_image中含有的目标 并保存为xml
                boxes_of_sub_img = [] # 子图中的框和名字 如果子图中没有 则不生成xml文件
                names_of_sub_img = []
                sub_img_shapely_cord = Polygon([
                    (left_up[0], left_up[1]+subsize-1), 
                    (left_up[0]+subsize-1, left_up[1]+subsize-1), 
                    (left_up[0]+subsize-1, left_up[1]), 
                    (left_up[0], left_up[1])
                ])
                # 遍历标注框
                for idx, bbox in enumerate(boxes):
                    bbox = Polygon(bbox)
                    # case1: 包含
                    if sub_img_shapely_cord.contains(bbox):
                        min_x, min_y, max_x, max_y = bbox.bounds
                        boxes_of_sub_img.append((int(min_x-left_up[0]), int(min_y-left_up[1]), int(max_x-left_up[0]), int(max_y-left_up[1])))
                        names_of_sub_img.append(names[idx])
                    # case2: 相交
                    elif sub_img_shapely_cord.intersects(bbox):
                        intersection = sub_img_shapely_cord.intersection(bbox)
                        # 只考虑相交形状为矩形的情况 为线和点的情况 相当于目标只占图像的一个像素或一排像素 不考虑
                        if intersection.type == "Polygon":
                            # 计算是否满足阈值的要求
                            area1 = intersection.area
                            area2 = bbox.area
                            if area1/area2 >= IoU_thresh:
                                min_x, min_y, max_x, max_y = intersection.bounds
                                boxes_of_sub_img.append((int(min_x-left_up[0]), int(min_y-left_up[1]), int(max_x-left_up[0]), int(max_y-left_up[1])))
                                names_of_sub_img.append(names[idx])
                    # case3: 目标大于等于子图
                    elif sub_img_shapely_cord.within(bbox):
                        boxes_of_sub_img.append((0, 0, subsize, subsize))
                        names_of_sub_img.append(names[idx])
                    # case4: disjoint 目标位于子图外 do nothing
                # 保存xml文件 同时根据要求是否删除空目标图像
                if len(boxes_of_sub_img)!=0:
                    create_voc_xml(
                        os.path.join(labels_save_path, "{}_{}_{}.xml".format(file_name, left_up[0], left_up[1])),
                        boxes_of_sub_img,
                        names_of_sub_img
                    )
                else:
                    if not keep_no_object_images:
                        os.remove(os.path.join(images_save_path, "{}_{}_{}.jpg".format(file_name, left_up[0], left_up[1])))
        # 打印单个图片完成处理的信息
        print(image, "finished!")


if __name__ == "__main__":
    ImagePath = "horizontal_object_detection/example_images"
    AnnotationPath = "horizontal_object_detection/example_labels"
    SavePath = "horizontal_object_detection/example_results"
    main(ImagePath, AnnotationPath, SavePath)



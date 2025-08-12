import os
import shutil
import json
from PIL import Image, UnidentifiedImageError
from collections import OrderedDict

def parse_txt(txt_path):
    """解析标注txt文件，返回标注列表"""
    ann_list = []
    with open(txt_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            if len(values) < 5:
                continue
            x, y, w, h = map(float, values[:4])
            category_id = int(values[4])
            ann_list.append({
                'bbox': [x, y, w, h],
                'category_id': category_id,
            })
    return ann_list

def images_and_annotations(img_dir, ann_dir):
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    res = []
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        txt_file = os.path.splitext(img_file)[0] + '.txt'
        txt_path = os.path.join(ann_dir, txt_file)
        res.append((img_file, img_path, txt_path))
    return res

def convert_split(src_img_dir, src_ann_dir, dst_split, categories, info, licenses):
    os.makedirs(dst_split, exist_ok=True)
    images, annotations = [], []
    ann_id, img_id = 1, 1
    skipped_images = 0
    for img_file, img_path, txt_path in images_and_annotations(src_img_dir, src_ann_dir):
        dst_img_path = os.path.join(dst_split, img_file)
        try:
            shutil.copy(img_path, dst_img_path)
            with Image.open(img_path) as img:
                width, height = img.size
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            print(f"警告: 图片无法处理或不存在，已跳过: {img_path}")
            skipped_images += 1
            continue

        images.append({
            'id': img_id,
            'file_name': img_file,
            'width': width,
            'height': height,
        })
        if os.path.exists(txt_path):
            anns = parse_txt(txt_path)
            for ann in anns:
                x, y, w, h = ann['bbox']
                area = w * h
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': ann['category_id'],
                    'bbox': [x, y, w, h],
                    'area': area,
                    'iscrowd': 0,
                })
                ann_id += 1
        img_id += 1
    # 使用OrderedDict保证字段顺序
    coco_json = OrderedDict([
        ('info', info),
        ('licenses', licenses),
        ('categories', categories),  # 现在在licenses之后
        ('images', images),
        ('annotations', annotations)
    ])
    with open(os.path.join(dst_split, '_annotations.coco.json'), 'w', encoding='utf-8') as f:
        json.dump(coco_json, f, ensure_ascii=False, indent=2)
    print(f"已生成:{os.path.join(dst_split, '_annotations.coco.json')}, 包含图片数: {len(images)}, 标注数: {len(annotations)}, 跳过图片: {skipped_images}")

def main(src_dirs, dst_root):
    info = {
        "description": "Converted COCO Format Dataset",
        "version": "1.0",
        "year": 2025,
        "contributor": "Unknown",
        "date_created": "2025-08-12"
    }
    licenses = [
        {
            "id": 1,
            "name": "Unknown",
            "url": ""
        }
    ]
    categories = [
        {"id": 0, "name": "pedestrian", "supercategory": "none"},
        {"id": 1, "name": "people", "supercategory": "none"},
        {"id": 2, "name": "bicycle", "supercategory": "none"},
        {"id": 3, "name": "car", "supercategory": "none"},
        {"id": 4, "name": "van", "supercategory": "none"},
        {"id": 5, "name": "truck", "supercategory": "none"},
        {"id": 6, "name": "tricycle", "supercategory": "none"},
        {"id": 7, "name": "awning-tricycle", "supercategory": "none"},
        {"id": 8, "name": "bus", "supercategory": "none"},
        {"id": 9, "name": "motor", "supercategory": "none"}
    ]
    dst_splits = ['train', 'valid', 'test']
    for i, src_dir in enumerate(src_dirs):
        src_img_dir = os.path.join(src_dir, 'images')
        src_ann_dir = os.path.join(src_dir, 'annotations')
        dst_split = os.path.join(dst_root, dst_splits[i])
        if os.path.exists(src_img_dir) and os.path.exists(src_ann_dir):
            convert_split(src_img_dir, src_ann_dir, dst_split, categories, info, licenses)
        else:
            print(f"跳过: {src_img_dir} 或 {src_ann_dir}，目录不存在")

if __name__ == '__main__':
    src_dirs = [
        '01_VisDrone/Task1_Object_Detection_in_Images/VisDrone2019-DET-train',
        '01_VisDrone/Task1_Object_Detection_in_Images/VisDrone2019-DET-val',
        '01_VisDrone/Task1_Object_Detection_in_Images/test',
    ]
    dst_root = '01_VisDrone_coco/Task1_Object_Detection_in_Images'
    main(src_dirs, dst_root)
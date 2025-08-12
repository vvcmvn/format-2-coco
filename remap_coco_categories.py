import os
import json

def remap_categories(coco_json_path, out_json_path, id_map, new_categories):
    """
    将COCO标注文件中的类别id按id_map映射为新id，并更新categories字段。
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 映射类别id
    for ann in coco['annotations']:
        old_id = ann['category_id']
        if old_id in id_map:
            ann['category_id'] = id_map[old_id]
        else:
            raise ValueError(f"标注中存在未知类别id: {old_id}")

    # 替换categories
    coco['categories'] = new_categories

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f'已处理: {out_json_path}')

def process_all_splits(data_root, id_map, new_categories, coco_json_name='_annotations.coco.json'):
    """
    自动处理 data_root 下的 train, val, test 子目录的 COCO 标注文件。
    """
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        coco_json_path = os.path.join(split_dir, coco_json_name)
        if os.path.exists(coco_json_path):
            out_json_path = os.path.join(split_dir, f'remapped_{coco_json_name}')
            remap_categories(coco_json_path, out_json_path, id_map, new_categories)
        else:
            print(f'跳过 {split_dir}，未找到标注文件：{coco_json_path}')

if __name__ == '__main__':
    # 示例：指定数据集根目录
    data_root = 'Pure_Tank_resized'
    # 例如，将id=0,2映射为1（tank），id=1映射为2（car）
    id_map = {0: 0, 1: 0}
    new_categories = [
        {"id": 0, "name": "tank", "supercategory": "none"},
    ]
    # 可指定COCO标注文件名，默认'_annotations.coco.json'
    process_all_splits(data_root, id_map, new_categories, coco_json_name='_annotations.coco.json')
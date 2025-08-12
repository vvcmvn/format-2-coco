import os
import json
from PIL import Image, ImageOps

def resize_and_pad(img, target_w, target_h):
    """Resize and pad image to target size, return new image and (scale_w, scale_h, pad_left, pad_top)"""
    orig_w, orig_h = img.size
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    scale = min(scale_w, scale_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    img_padded = ImageOps.expand(img_resized, border=(pad_left, pad_top, target_w-new_w-pad_left, target_h-new_h-pad_top), fill=(0,0,0))
    return img_padded, scale, pad_left, pad_top

def process_split(split_in_dir, split_out_dir, json_name='_annotations.coco.json', target_w=800, target_h=1440):
    os.makedirs(split_out_dir, exist_ok=True)

    # 1. 处理图片
    with open(os.path.join(split_in_dir, json_name), 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_id_map = {}
    for img_info in data['images']:
        fname = img_info['file_name']
        img_path = os.path.join(split_in_dir, fname)
        img = Image.open(img_path)
        img_padded, scale, pad_left, pad_top = resize_and_pad(img, target_w, target_h)
        img_padded.save(os.path.join(split_out_dir, fname))   # 保持原名
        img_id_map[img_info['id']] = (scale, pad_left, pad_top, img.size)
        # 修改json里的宽高
        img_info['width'] = target_w
        img_info['height'] = target_h
        img_info['file_name'] = fname  # 保持原名

    # 2. 修改标注
    for ann in data['annotations']:
        scale, pad_left, pad_top, (orig_w, orig_h) = img_id_map[ann['image_id']]
        # bbox: [x, y, w, h]
        x, y, w, h = ann['bbox']
        x = x * scale + pad_left
        y = y * scale + pad_top
        w = w * scale
        h = h * scale
        ann['bbox'] = [x, y, w, h]
        if 'area' in ann:
            ann['area'] = ann['area'] * scale * scale
        # segmentation
        if 'segmentation' in ann and ann['segmentation']:
            new_segs = []
            for seg in ann['segmentation']:
                new_seg = []
                for i in range(0, len(seg), 2):
                    pt_x = seg[i] * scale + pad_left
                    pt_y = seg[i+1] * scale + pad_top
                    new_seg += [pt_x, pt_y]
                new_segs.append(new_seg)
            ann['segmentation'] = new_segs

    # 3. 保存新json
    out_json = os.path.join(split_out_dir, json_name)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'{split_out_dir}: Done, new images and json saved.')

# 用法示例
def main():
    in_root = 'A022_训练数据/01_VisDrone_coco/Task1_Object_Detection_in_Images'
    out_root = 'A022_训练数据/01_VisDrone_coco/Task1_Object_Detection_in_Images_resized'
    splits = ['train', 'valid', 'test']
    for split in splits:
        process_split(
            os.path.join(in_root, split),
            os.path.join(out_root, split),
            json_name='_annotations.coco.json',
            target_w=800,
            target_h=1440
        )

if __name__ == "__main__":
    main()
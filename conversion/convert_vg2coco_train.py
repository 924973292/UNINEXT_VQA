import json
import argparse
import pycocotools.mask as maskUtils
from detectron2.structures import PolygonMasks


def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--src_json",
                        default="/13994058190/WYH/UNINEXT/datasets/VG/question_answers.json",
                        type=str, help="the original json file")
    parser.add_argument("--des_json",
                        default="/13994058190/WYH/UNINEXT/datasets/VG/question_answers.json",
                        type=str, help="the processed json file")
    return parser.parse_args()


def compute_area(segmentation):
    if isinstance(segmentation, list):
        polygons = PolygonMasks([segmentation])
        area = polygons.area()[0].item()
    elif isinstance(segmentation, dict):  # RLE
        area = maskUtils.area(segmentation).item()
    else:
        raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
    return area


if __name__ == "__main__":
    args = parse_args()
    data = json.load(open(args.src_json, 'r'))
    new_data = {"images": [], "annotations": [],
                "categories": [{"supercategory": "object", "id": 1, "name": "object"}]}
    inst_idx = 0  # index of the instance
    for i in range(len(data)):
        item_data = data[i]
        image_id = item_data['id']
        for j in range(len(item_data['qas'])):
            question_data = item_data['qas'][j]
            inst_idx += 1
            image = {"file_name": "%d.jpg" % image_id, "height": 420, "width": 420, 'id': inst_idx}
            image["expressions"] = []
            image["expressions"].append(question_data['question']+' '+question_data['answer'])
            anno = {"bbox": [0, 0, 420, 420], "image_id": inst_idx, \
                    "iscrowd": 0, "category_id": 1, "id": inst_idx}
            new_data["images"].append(image)
            new_data["annotations"].append(anno)
        if len(item_data['qas']) == 0:
            inst_idx += 1
            image = {"file_name": "%d.jpg" % image_id, "height": 420, "width": 420, 'id': inst_idx}
            image["expressions"] = []
            image["expressions"].append("What is this?"+" "+"Image")
            anno = {"bbox": [0, 0, 420, 420], "image_id": inst_idx, \
                    "iscrowd": 0, "category_id": 1, "id": inst_idx}
            new_data["images"].append(image)
            new_data["annotations"].append(anno)
    output_json = args.des_json.replace(".json", "_train.json")
    json.dump(new_data, open(output_json, 'w'))

import json
import argparse
import pycocotools.mask as maskUtils
from detectron2.structures import PolygonMasks


def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--src_json",
                        default="/13994058190/WYH/UNINEXT/datasets/VQAv2/v2_mscoco_train2014_annotations.json",
                        type=str, help="the original json file")
    parser.add_argument("--src_questions",
                        default="/13994058190/WYH/UNINEXT/datasets/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json",
                        type=str, help="the original json file")
    parser.add_argument("--des_json",
                        default="/13994058190/WYH/UNINEXT/datasets/VQAv2/v2_mscoco_train2014_annotations.json",
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
    questions = json.load(open(args.src_questions, 'r'))
    questions_all = questions["questions"]
    inst_idx = 0  # index of the instance
    new_data = {"images": [], "annotations": [],
                "categories": [{"supercategory": "object", "id": 1, "name": "object"}]}
    index = 0
    for split in data.keys():
        if split == "annotations":
            for cur_data in data[split]:
                cur_question = questions_all[index]['question']
                index += 1
                for entry in cur_data["answers"]:
                    inst_idx += 1
                    image = {"file_name": "COCO_train2014_%012d.jpg" % cur_data["image_id"], "height": 420,
                             "width": 420, \
                             "id": inst_idx}
                    image["expressions"] = [cur_question+' '+entry["answer"]]
                    anno = {"bbox": [0, 0, 420, 420], "image_id": inst_idx, \
                            "iscrowd": 0, "category_id": 1, "id": inst_idx}
                    new_data["images"].append(image)
                    new_data["annotations"].append(anno)
                    print(inst_idx)
            output_json = args.des_json.replace(".json", "_show.json")
            json.dump(new_data, open(output_json, 'w'))

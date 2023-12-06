import json
import argparse
import pycocotools.mask as maskUtils
from detectron2.structures import PolygonMasks


def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--src_json",
                        default="/13994058190/WYH/UNINEXT/datasets/annotations/referitgame-berkeley/instances.json", type=str,
                        help="the original json file")
    parser.add_argument("--des_json",
                        default="/13994058190/WYH/UNINEXT/datasets/annotations/referitgame-berkeley/instances.json", type=str,
                        help="the processed json file")
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


# if __name__ == "__main__":
#     args = parse_args()
#     data = json.load(open(args.src_json, 'r'))
#     inst_idx = 0  # index of the instance
#     new_data = {"images": [], "annotations": [], "categories": [{"supercategory": "object", "id": 1, "name": "object"}]}
#     for split in data.keys():
#         for cur_data in data[split]:
#             for expression in cur_data["expressions"]:
#                 inst_idx += 1
#                 image = {"file_name": "COCO_train2014_%012d.jpg" % cur_data["image_id"], "height": cur_data["height"],
#                          "width": cur_data["width"], \
#                          "id": inst_idx, }
#                 image["expressions"] = [expression]
#                 area = compute_area(cur_data["mask"])
#                 anno = {"bbox": cur_data["bbox"], "segmentation": cur_data["mask"], "image_id": inst_idx, \
#                         "iscrowd": 0, "category_id": 1, "id": inst_idx, "area": area}
#                 new_data["images"].append(image)
#                 new_data["annotations"].append(anno)
#
#     # 检查output_json是否存在,若不存在就创建
#     import os
#
#     if not os.path.exists(args.des_json):
#         os.makedirs(args.des_json)
#
#     output_json = args.des_json.replace(".json", "_show.json")
#     json.dump(new_data, open(output_json, 'w'))


if __name__ == "__main__":
    args = parse_args()
    data = json.load(open(args.src_json, 'r'))
    inst_idx = 0  # index of the instance
    new_data = {"images": [], "annotations": [], "categories": [{"supercategory": "object", "id": 1, "name": "object"}]}
    for split in data.keys():
        for cur_data in data[split]:
            for expression in cur_data["expressions"]:
                inst_idx += 1
                image = {"file_name": "%d.jpg" % cur_data["image_id"], "height": cur_data["height"],
                         "width": cur_data["width"], \
                         "id": inst_idx, "expressions": [expression]}
                anno = {"bbox": cur_data["bbox"], "image_id": inst_idx, "iscrowd": 0, "category_id": 1, "id": inst_idx}

                new_data["images"].append(image)
                new_data["annotations"].append(anno)

    # 检查output_json是否存在,若不存在就创建
    import os

    if not os.path.exists(args.des_json):
        os.makedirs(args.des_json)

    output_json = args.des_json.replace(".json", "_show.json")
    json.dump(new_data, open(output_json, 'w'))

import argparse
import os
import hashlib
import numpy as np
import json
import torch
import torchvision
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import groundingdino.datasets.transforms as T
import torchvision.transforms as TS

import base64
from io import BytesIO

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import pycocotools.mask as mask_util

from COCOStuff import CocoStuffBboxCaptionDatasetPaddingVersion

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def encode_tensor_as_string(arr):
    if type(arr) != np.ndarray:
        arr = arr.data.cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')

# load BLIP and CLIP model

def load_qwen(device):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True, fp16=True).eval()
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, fp16=True).eval().to(device)
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    return model, tokenizer

# generate caption using BLIP and get the CLIP embeddings
def forward_qwen(raw_image, category_name, bbox, qwen_model, qwen_tokenizer):
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    ### get instance caption using BLIP
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) # bbox = (left, top, right, bottom)
    if area >= 32*32:
        # crop the image using bbox
        raw_image_cropped = raw_image.crop(bbox)
        # resize the image to smaller side = 512 while maintaining the aspect ratio
        width, height = raw_image_cropped.size
        if height > width:
            raw_image_cropped = raw_image_cropped.resize((512, int(512*height/width)))
        else:
            raw_image_cropped = raw_image_cropped.resize((int(512*width/height), 512))

        tmp_image_name = hashlib.md5(raw_image_cropped.tobytes()).hexdigest()
        tmp_image_name = os.path.join('tmp', tmp_image_name + '.jpg')
        os.makedirs('tmp', exist_ok=True)
        raw_image_cropped.save(tmp_image_name)

        category_name = category_name.replace('-other', '')

        query = qwen_tokenizer.from_list_format([
            {
                'image': tmp_image_name,
            },
            {
                # 'text': 'You are viewing an image. Please describe the content of the image in one sentence, focusing specifically on the spatial relationships between objects. Include detailed observations about all the objects and how they are positioned in relation to other objects in the image. Your response should be limited to this description, without any additional information'
                'text': f'You are viewing an image with the main subject {category_name}. Please describe the content of the image in one sentence, focusing specifically on the spatial relationships between objects, and the main subject of the description should be {category_name}. Include detailed observations about all the objects and how they are positioned in relation to other objects in the image. Your response should be limited to this description, without any additional information'
            }
        ])
        response, history = qwen_model.chat(
            qwen_tokenizer, query=query, history=None
        )
        instance_caption = response
        # remove the temporary image
        os.remove(tmp_image_name)
        # print('Object size: {}; Instance caption: {}; category_name: {}'.format(area, instance_caption, category_name))

    else:
        instance_caption = category_name
    
    return instance_caption


def save_mask_data(output_dir, box_list, label_list, file_name, image_pil, output, qwen_model, qwen_tokenizer):
    value = 0  # 0 for background

    for label, box in zip(label_list, box_list):
        value += 1
        name = label
        logit = 1. # this is gt coco label, so set to 1

        box_xywh = [int(x) for x in box.tolist()]
        box_xywh[2] = box_xywh[2] - box_xywh[0]
        box_xywh[3] = box_xywh[3] - box_xywh[1]

        anno = get_base_anno_dict(is_stuff=0, is_thing=1, bbox=box_xywh, pred_score=float(logit), mask_value=value, rle=None, category_name=name, area=box_xywh[-1]*box_xywh[-2])
        RGB_image = image_pil.convert('RGB')
        x1y1x2y2 = [int(x) for x in box.tolist()]
        instance_caption = forward_qwen(RGB_image, name, x1y1x2y2, qwen_model, qwen_tokenizer)
        anno['caption'] = instance_caption
        output['annos'].append(anno)

    with open(os.path.join(output_dir, 'label_{}.json'.format(file_name)), 'w') as f:
        json.dump(output, f, indent=4)
        print("Saved {}/label_{}.json".format(output_dir, file_name))

# convert PIL image to base64
def encode_pillow_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_base_output_dict(image, dataset_name, image_path, data=None):
    output = {}
    if data != None:
        if 'similarity' in data:
            output['similarity'] = data['similarity']
        if 'AESTHETIC_SCORE' in data:
            output['AESTHETIC_SCORE'] = data['AESTHETIC_SCORE']
        if 'caption' in data:
            output['caption'] = data['caption']
        if 'width' in data:
            output['width'] = data['width']
        if 'height' in data:
            output['height'] = data['height']
        if 'file_name' in data:
            output['file_name'] = data['file_name']
        if 'is_det' in data:
            output['is_det'] = data['is_det']
        else:
            output['is_det'] = 0
        if 'image' in data:
            output['image'] = data['image']
    else:
        output['file_name'] = image_path # image_paths[i]
        output['is_det'] = 1
        output['image'] = encode_pillow_to_base64(image.convert('RGB'))
    output['dataset_name'] = dataset_name
    output['data_id'] = 1
    # annos for all instances
    output['annos'] = []
    return output

def get_base_anno_dict(is_stuff, is_thing, bbox, pred_score, mask_value, rle, category_name, area):
    anno = {
        "id": 0,
        "isfake": 0,
        "isreflected": 0,
        "bbox": bbox,
        "mask_value": mask_value,
        "mask": rle,
        "pred_score": pred_score,
        "category_id": 0,
        "data_id": 0,
        "category_name": category_name,
        "text_embedding_before": "",
        "caption": "",
        "blip_clip_embeddings": "",
        "is_stuff": is_stuff,
        "is_thing": is_thing,
        "area": area
    }
    return anno

def get_args_parser():
    parser = argparse.ArgumentParser('Caption Generation script', add_help=False)

    parser.add_argument("--split", default="val", type=str, help="split for coco")
    parser.add_argument("--image_size", type=int, default=768, help="image size for model inference")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    return parser

def main(args):

    # cfg
    output_dir = args.output_dir
    device = 'cuda'
    
    # load qwen model
    print("Initialize Qwen model")
    qwen_model, qwen_tokenizer = load_qwen(device)

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    dataset = CocoStuffBboxCaptionDatasetPaddingVersion(
        root='/home/ubuntu/DetailedSD/data/coco_2017/',
        image_size=args.image_size,
        validation=True if args.split == 'val' else False,
        min_objects_per_image=1,
        max_objects_per_image=100,
        stuff_only=True,
    )

    # load images for model inference
    num_images = 0
    dataset_name = 'rich-context-coco'
    # read image and captions from json file

    # iterate over all images
    for batch in dataset:
        image_path = batch['image_path']
        image_caption = batch['caption']
        bboxes = batch['bboxes']
        labels = batch['labels']
        image_th = batch['image']
        image_pil = TS.ToPILImage()(image_th)
        img_meta_data = {} # store image meta data
        # dataset = wds.WebDataset([tar]).decode("pil")
        img_name_base = image_path.split("/")[-1].split(".")[0]

        # save the image caption
        img_meta_data['caption'] = image_caption
        
        # save file name
        file_name = img_name_base
        img_meta_data['file_name'] = image_path

        if os.path.exists(os.path.join(output_dir, 'label_{}.json'.format(file_name))):
            print("SKIPPING: Found processed {} image; {}".format(num_images, file_name))
            continue

        # get base output dictionary
        output = get_base_output_dict(image_pil, dataset_name, file_name, data=img_meta_data)

        num_images += 1

        # save mask data
        save_mask_data(output_dir, bboxes, labels, file_name, image_pil, output, qwen_model, qwen_tokenizer)
        print("Processed {} image; {}".format(num_images, file_name))

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
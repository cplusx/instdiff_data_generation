import argparse
import os
import hashlib
import numpy as np
import json
import torch
import torchvision
from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    SamPredictor
) 
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
import sys
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

import base64
from io import BytesIO

# import openai
# import nltk
# from lavis.models import load_model_and_preprocess
# from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import pycocotools.mask as mask_util

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

        # raw_image_cropped = blip_vis_processors["eval"](raw_image_cropped).unsqueeze(0).to(device)
        # generate caption using beam search
        # captions = blip_model.generate({"image": raw_image_cropped})
        # instance_caption = captions[0]
        # instance_caption_all = captions[0]
        # print('Instance caption: {}; category_name: {}'.format(captions, category_name))
        query = qwen_tokenizer.from_list_format([
            {
                'image': tmp_image_name,
            },
            {
                'text': 'You are viewing an image. Please describe the content of the image in one sentence, focusing specifically on the spatial relationships between objects. Include detailed observations about all the objects and how they are positioned in relation to other objects in the image. Your response should be limited to this description, without any additional information'
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

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

# decode base64 to pillow image
def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

# read images
def read_image_from_json(filename):
    # read RGB image with Image
    with open(filename, 'r') as f:
        data = json.load(f)
    img = decode_base64_to_pillow(data['image'])
    return img, data

# apply image transformation
def apply_img_transform(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cuda"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

# convert binary mask to RLE
def mask_2_rle(binary_mask):
    # binary_mask_encoded = mask_pycoco.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # area = mask_pycoco.area(binary_mask_encoded)
    # if area < 16:
    #     return None, None
    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def save_mask_data(output_dir, mask_list, box_list, label_list, file_name, image_pil, output, qwen_model, qwen_tokenizer):
    value = 0  # 0 for background

    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'

        mask = mask_list[value-1].cpu().numpy()[0] == True
        rle = mask_2_rle(mask)

        box_xywh = [int(x) for x in box.numpy().tolist()]
        box_xywh[2] = box_xywh[2] - box_xywh[0]
        box_xywh[3] = box_xywh[3] - box_xywh[1]

        anno = get_base_anno_dict(is_stuff=0, is_thing=1, bbox=box_xywh, pred_score=float(logit), mask_value=value, rle=rle, category_name=name, area=box_xywh[-1]*box_xywh[-2])
        RGB_image = image_pil.convert('RGB')
        x1y1x2y2 = [int(x) for x in box.numpy().tolist()]
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
    parser.add_argument("--job_index", type=int, default=0, help="")
    parser.add_argument("--num_jobs", type=int, default=1, help="")

    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--image_folder", type=str, default=None, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--train_data_path", type=str, required=True, help="path to the json file with image path and image caption")

    return parser

def main(args):

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device1 = 'cuda'
    device2 = 'cuda'
    
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device2)
    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])

    # load model
    # ram_model = tag2text.ram(pretrained=ram_checkpoint,
    ram_model = ram(pretrained=ram_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()
    ram_model = ram_model.to(device2)

    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        from segment_anything_hq import sam_model_registry
        model_type = "vit_h"
        predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_hq_checkpoint).to(device2))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device2))
    
    # load BLIP and CLIP model
    # print("Initialize BLIP and CLIP model")
    # blip_model, blip_vis_processors, clip_text_model, clip_text_processor = load_blip_clip(device1)
    # load qwen model
    print("Initialize Qwen model")
    qwen_model, qwen_tokenizer = load_qwen(device1)

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load images for model inference
    num_images = 0
    dataset_name = 'ss-sim-0.3-aesthetic-5'
    # read image and captions from json file
    json_path = args.train_data_path
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_paths = []
    image_captions = []
    for i in range(len(data)):
        image_path = data[i]['image']
        if args.image_folder:
            image_batch_folder, image_name = image_path.split('/')[-2:]
            image_path = os.path.join(args.image_folder, image_batch_folder, image_name)
        image_paths.append(image_path)
        image_captions.append(data[i]['caption'])
    # print(image_paths)

    # split the image list into multiple jobs
    n_images = len(image_paths)
    print('Overall number of images: ', n_images)
    num_imgs_per_job = n_images // args.num_jobs + 1
    start_idx = args.job_index * num_imgs_per_job
    end_idx = min((args.job_index + 1) * num_imgs_per_job, n_images)
    print("start_idx, end_idx", start_idx, end_idx)

    # iterate over all images
    for image_path, image_caption in zip(image_paths[start_idx:end_idx], image_captions[start_idx:end_idx]):
        img_meta_data = {} # store image meta data
        # dataset = wds.WebDataset([tar]).decode("pil")
        img_name_base = image_path.split("/")[-1].split(".")[0]

        # read image and convert to RGB image using PIL
        image_pil = Image.open(image_path).convert("RGB")  # load image

        # save raw image
        # img_meta_data['image'] = encode_pillow_to_base64(image_pil.convert('RGB'))

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
        image = apply_img_transform(image_pil)

        # # visualize raw image
        # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

        # run RAM model
        raw_image = image_pil.resize((384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(device2)
        res = inference_ram(raw_image , ram_model)

        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        tags=res[0].replace(' |', ',')

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, tags, box_threshold, text_threshold, device=device2
        )
        image = np.array(image_pil)
        predictor.set_image(image)
        
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()

        # use NMS to handle overlapped boxes
        # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        # print(f"After NMS: {boxes_filt.shape[0]} boxes")

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device2)

        try:
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device2),
                multimask_output = False,
            )
        except:
            continue

        num_images += 1

        # save mask data
        save_mask_data(output_dir, masks, boxes_filt, pred_phrases, file_name, image_pil, output, qwen_model, qwen_tokenizer)
        print("Processed {} image; {}".format(num_images, file_name))

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
import os
import json
import datetime as dt
import requests
import numpy as np
from PIL import Image
import imageio
from sklearn import model_selection

from pycocotools.coco import COCO
from tqdm import tqdm


JSON_PATH = './data/annotations/coco_final.json'
MAKE_JSON_PATH = './data/annotations/instances_{}2017.json'

OUTPUT_PATH = './data'

# SEG_PATH_FULL = './data/segmentation_full'
# SEG_PATH_SUB = './data/segmentation_sub'

# os.makedirs(SEG_PATH_FULL, exist_ok=True)
# os.makedirs(SEG_PATH_SUB, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
# os.makedirs(os.path.join(SEG_PATH_FULL, 'images'), exist_ok=True)
# os.makedirs(os.path.join(SEG_PATH_FULL, 'masks'), exist_ok=True)
# os.makedirs(os.path.join(SEG_PATH_SUB, 'images'), exist_ok=True)
# os.makedirs(os.path.join(SEG_PATH_SUB, 'masks'), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_PATH, 'images'), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_PATH, 'labels'), exist_ok=True)

def process_data(images, data_type="train"):
    # os.makedirs(os.path.join(SEG_PATH_FULL, f'images/{data_type}/'), exist_ok=True)
    # os.makedirs(os.path.join(SEG_PATH_FULL, f'masks/{data_type}/'), exist_ok=True)
    # os.makedirs(os.path.join(SEG_PATH_SUB, f'images/{data_type}/'), exist_ok=True)
    # os.makedirs(os.path.join(SEG_PATH_SUB, f'masks/{data_type}/'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, f'{data_type}2017/'), exist_ok=True)
    # os.makedirs(os.path.join(OUTPUT_PATH, f'labels/{data_type}/'), exist_ok=True)
    
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)  
    # info = coco.info 

    # setup COCO dataset container and info
    coco_ = {
        'info': None,
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': []
    }

    coco_['info'] = {
        'year': dt.datetime.now(dt.timezone.utc).year,
        'version': None,
        'description': None,
        'contributor': None,
        'url': None,
        'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
    }  
        
    coco_['categories'] = cats

    for im in tqdm(images, total=len(images)):

        img = imageio.imread(im['coco_url'])
        
        annIds = coco.getAnnIds(imgIds=[im['id']])
        anns = coco.loadAnns(annIds)

        image_name = im['file_name']
        
        image = {
            "id": im["id"],
            "width": im["width"],
            "height": im["height"],
            "file_name": image_name,
            "license": im["license"],
            "flickr_url": im["flickr_url"],
            "coco_url": im["coco_url"],
            "date_captured": im["date_captured"]
        }

        coco_['images'].append(image)
        
        #save iamges
        imageio.imwrite(os.path.join(OUTPUT_PATH, f'{data_type}2017/{image_name}'), img)

        # convert Labelbox Polygon to COCO Polygon format
        for i in range(len(anns)):

            annotation = {
                "id": len(coco_['annotations']) + 1,
                "image_id": anns[i]["image_id"],
                "category_id": anns[i]["category_id"],
                "segmentation": anns[i]["segmentation"] ,
                "area": round(float(anns[i]["area"]), 3),  # float
                "bbox": [round(float(point), 3) for point in anns[i]["bbox"]],
                "iscrowd": anns[i]["iscrowd"]
            }

            coco_['annotations'].append(annotation)

    with open(MAKE_JSON_PATH.format(data_type), 'w+') as f:
        f.write(json.dumps(coco_))
        
        
        
        
        
if __name__ == "__main__":
    coco = COCO(JSON_PATH)
    print(coco)
    
    # # get all category names
    # cats = coco.loadCats(coco.getCatIds())
    # cats = [cat['name'] for cat in cats]
    
    # get all ImgIds and images
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)

    # batch size
    batch_size = 32
    
    # train, validation split
    # train and validation can be divided by batch size (train:validation=8:2)
    train, valid = model_selection.train_test_split(
        images,
        train_size=len(images)//(5*batch_size)*(4*batch_size),
        test_size=len(images)//(5*batch_size)*(1*batch_size),
        random_state=42,
        shuffle=True        
    )
    
    process_data(train, data_type="train")
    process_data(valid, data_type="val")

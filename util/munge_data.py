import os
import requests
import numpy as np
from PIL import Image
import imageio
from sklearn import model_selection

from pycocotools.coco import COCO
from tqdm import tqdm


JSON_PATH_TRAIN = './data/annotations/instances_train2017.json'
JSON_PATH_VAL = './data/annotations/instances_val2017.json'

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
    for im in tqdm(images, total=len(images)):

        img = imageio.imread(im['coco_url'])
        
        annIds = coco.getAnnIds(imgIds=[im['id']])
        anns = coco.loadAnns(annIds)
        
        image_name = im['file_name']
        # filename = image_name.replace("JPG", "txt")
        width = im['width']
        height = im['height']
        
        # #yolo data
        
        # yolo_data = []
        # for i in range(len(anns)):
        #     cat = anns[i]["category_id"] - 1
        #     xmin = anns[i]["bbox"][0]
        #     ymin = anns[i]["bbox"][1]
        #     xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
        #     ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

        #     x = (xmin + xmax)/2
        #     y = (ymin + ymax)/2

        #     w = xmax - xmin
        #     h = ymax - ymin

        #     x /= width
        #     w /= width
        #     y /= height
        #     h /= height
            
        #     yolo_data.append([cat, x, y, w, h])
            
        # yolo_data = np.array(yolo_data)
        
        # os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # # save labels
        # np.savetxt(
        #     os.path.join(OUTPUT_PATH, f"labels/{data_type}/{filename}"),
        #     yolo_data,
        #     fmt=["%d", "%f", "%f", "%f", "%f"]
        # )
        
        #save iamges
        imageio.imwrite(os.path.join(OUTPUT_PATH, f'{data_type}2017/{image_name}'), img)
        
        
        # deeplab per bbox version
        # for i in range(len(anns)):
        #     xmin = anns[i]["bbox"][0]
        #     ymin = anns[i]["bbox"][1]
        #     xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
        #     ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

        #     base = 50
        #     xmin = xmin - base if (xmin - base) >= 0 else 0
        #     ymin = ymin - base if (ymin - base) >= 0 else 0
        #     xmax = xmax + base if (xmax + base) <= 1500 else 1500
        #     ymax = ymax + base if (ymax + base) <= 1500 else 1500
            
        #     sub_image = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
        #     mask = coco.annToMask(anns[i]) * 255.
        #     mask = mask[int(ymin):int(ymax), int(xmin):int(xmax)]
        #     #save images
        #     imageio.imwrite(os.path.join(SEG_PATH_SUB, f'images/{data_type}/{image_name[:-4]}_{i}.jpg'), sub_image.astype(np.uint8))
        #     #save masks
        #     imageio.imwrite(os.path.join(SEG_PATH_SUB, f'masks/{data_type}/{image_name[:-4]}_{i}.jpg'), mask.astype(np.uint8))


        # deeplab full img version
        # mask_all = np.zeros((height, width))
        # for i in range(len(anns)):
        #     xmin = anns[i]["bbox"][0]
        #     ymin = anns[i]["bbox"][1]
        #     xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
        #     ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

            
        #     mask = coco.annToMask(anns[i]) * 255.
            
        #     mask_all[int(ymin):int(ymax), int(xmin):int(xmax)] = mask[int(ymin):int(ymax), int(xmin):int(xmax)]
            
        # #save images
        # imageio.imwrite(os.path.join(SEG_PATH_FULL, f'images/{data_type}/{image_name}'), img.astype(np.uint8))
        # #save masks
        # imageio.imwrite(os.path.join(SEG_PATH_FULL, f'masks/{data_type}/{image_name}'), mask_all.astype(np.uint8))

        
        
        
        
        
if __name__ == "__main__":
    # make train data
    coco_train = COCO(JSON_PATH_TRAIN)
    print(coco_train.info)
    # get all ImgIds and images
    imgIds_train = coco_train.getImgIds()
    train = coco_train.loadImgs(imgIds_train)
    process_data(train, data_type="train")

    # make validation data
    coco_val = COCO(JSON_PATH_VAL)
    print(coco_val.info)
    # get all ImgIds and images
    imgIds_val = coco_val.getImgIds()
    valid = coco_val.loadImgs(imgIds_val)
    process_data(valid, data_type="val")

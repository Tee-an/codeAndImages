import os
import json
import numpy as np
import cv2
from labelme.utils import shape_to_mask

def json2PNG(fileDir):
    files = [i for i in os.listdir(fileDir) if i.endswith(".json")]
    print(len(files))

    sample = 0
    for i in range(len(files)):
        with open(fileDir + "/" +files[i]) as f:
            data = json.load(f)
        image_height,image_width = data['imageHeight'],data['imageWidth']
        sample += len(data['shapes'])
        mask = np.zeros((image_height,image_width),dtype=np.uint8)
        for shape in data['shapes']:
            points = shape['points']
            label = shape['label']
            shapeType = shape.get('shape_type','polygon')
            if label == 'TARs':
                mask_shape = shape_to_mask((image_height,image_width),points,shapeType)
                mask += mask_shape.astype(np.uint8)
        mask = np.clip(mask,0,1)
        cv2.imwrite(fileDir + "-ground" + "/" + files[i] + "-mask.png",mask * 255)
    print(sample)

if __name__ == "__main__":
    fileDir = "../data/ground/1"
    json2PNG(fileDir)



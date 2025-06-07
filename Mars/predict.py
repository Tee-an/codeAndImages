import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
from UNet import UNet


def pad_to_square(img,target_size=512):
    h,w = img.shape[:2]
    pad_h = max(target_size - h,0)
    pad_w = max(target_size - w,0)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        img,top,bottom,left,right,
        cv2.BORDER_REFLECT_101
    )

    return padded,(top,bottom,left,right)


def predict(image_path,model_path,output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

    image = cv2.imread(image_path)
    original_size = image.shape[:2]

    padding_img,padding_info = pad_to_square(image)
    img_tensor = torch.from_numpy(padding_img).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    top,bottom,left,right = padding_info
    h,w = original_size
    final_mask = binary_mask[top:top+h,left:left+w]

    file_name = (image_path.split('/')[3]).split('.tif')[0]
    cv2.imwrite(output_path + file_name + "-mask.png",final_mask)


if __name__ == '__main__':
    imageDir = "../data/output3"
    images = [i for i in os.listdir(imageDir) if i.endswith('.tif')]

    model_path = "../models/best_model.pth"
    output_path = "../data/prediction3/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i in tqdm(range(len(images))):
        images_path = imageDir + '/' + images[i]
        predict(images_path,model_path,output_path)

    print("任务完成!")
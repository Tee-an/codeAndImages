import os
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import albumentations as A


class DataSet(Dataset):
    def __init__(self,image_paths,mask_paths,transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx],cv2.IMREAD_GRAYSCALE)

        if image.shape[:2] != (512,512) or mask.shape != (512,512):
            print("尺寸不匹配！")
            return -1

        if self.transform:
            augumented = self.transform(image=image,mask=mask)
            image = augumented['image']
            mask = augumented['mask']

        image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        return image,mask


train_transform = A.Compose([
    A.Rotate(limit=20,p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(p=0.3)
])


def loadData(dataDir,test_size = 0.2):
    images_paths = sorted([dataDir + "/" + f for f in os.listdir(dataDir) if f.endswith('.tif')])
    mask_paths = sorted([dataDir + "/" + f for f in os.listdir(dataDir) if f.endswith('.png')])

    del_list = []
    for image in images_paths:
        tmp_mask = image.split(".tif")[0] + ".json-mask.png"
        if tmp_mask not in mask_paths:
            del_list.append(image)

    for image in del_list:
        images_paths.remove(image)

    assert len(images_paths) == len(mask_paths), "图像和掩膜数量不匹配,删除操作失败"

    train_img,val_img,train_mask,val_mask = train_test_split(
        images_paths,mask_paths,test_size=test_size,random_state=42
    )

    train_dataset = DataSet(train_img,train_mask,transform=train_transform)
    val_dataset = DataSet(val_img,val_mask)

    batch_size = 8
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,pin_memory=True)

    return train_loader,val_loader


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self,n_channels=3,n_classes=1):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)

        self.up1 = Up(512,256)
        self.up2 = Up(256,128)
        self.up3 = Up(128,64)
        self.outc = OutConv(64,n_classes)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)
        logits = self.outc(x)
        return logits


class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)


# class AttentionBlock(nn.Module):
#     def __init__(self,F_g,F_l):
#         super().__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g,F_l,kernel_size=1),
#             nn.BatchNorm2d(F_l)
#         )
#         self.W_x = nn.Conv2d(F_l,F_l,kernel_size=1)
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_l,1,kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self,g,x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = F.relu(g1+x1)
#         psi = self.psi(psi)
#         return x * psi


class CBMA(nn.Module):
    def __init__(self,in_channels,reduction_ratio=4):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,in_channels // reduction_ratio,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio,in_channels,kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2,1,kernel_size=7,padding=3),
            nn.Sigmoid()
        )

    def forward(self,x):
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights

        max_pool = torch.max(x_channel,dim=1,keepdim=True)[0]
        avg_pool = torch.mean(x_channel,dim=1,keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([max_pool,avg_pool],dim=1))
        x_spatial = x_channel * spatial_weights

        return x_spatial

class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)
        self.cbma = CBMA(in_channels // 2)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x2 = self.cbma(x2)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1,[diffX // 2,diffX - diffX // 2,
                       diffY // 2,diffY - diffY // 2])
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self,x):
        return self.conv(x)

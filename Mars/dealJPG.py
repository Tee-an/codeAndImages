from osgeo import gdal
import os
from tqdm import tqdm
import numpy as np

gdal.UseExceptions()

fileDir = r"D:\python\data\input3"
outPath = r"D:\python\data\output3"
if not os.path.exists(outPath):
    os.mkdir(outPath,0o774)

files = [i for i in os.listdir(fileDir) if i.endswith(".JP2")]
print(files)

size = 512
for img in range(len(files)):
    in_ds = gdal.Open(fileDir+"\\"+files[img])

    width = in_ds.RasterXSize
    height = in_ds.RasterYSize
    outbandsize = in_ds.RasterCount
    print(outbandsize)
    im_geotrans = in_ds.GetGeoTransform()
    im_proj = in_ds.GetProjection()

    col_num = int(width / size)
    row_num = int(height / size)
    if(width % size != 0):
        col_num += 1
    if(height % size != 0):
        row_num += 1

    print("width = {},height = {},bands = {},projection = {}\n,col_num = {},row_num = {}".format(width,
                                                                                                  height,
                                                                                                  outbandsize,
                                                                                                  im_proj,
                                                                                                  col_num,
                                                                                                  row_num))
    for i in tqdm(range(row_num)):
        for j in range(col_num):
            offset_x = j * size
            offset_y = i * size

            b_xsize = min(width - offset_x,size)
            b_ysize = min(height - offset_y,size)

            im_data = in_ds.GetRasterBand(1).ReadAsArray(offset_x,offset_y,b_xsize,b_ysize)
            im_data_normalized = ((im_data - np.min(im_data)) / (np.max(im_data) - np.min(im_data)) * 255).astype(np.uint8)

            # zeroNum = 0
            # for a in range(b_ysize):
            #     for b in range(b_xsize):
            #         if im_data[a][b] == 0:
            #             zeroNum += 1
            # if zeroNum / (size * size) > 0.8:
            #     continue

            tif_driver = gdal.GetDriverByName("GTiff")
            file = outPath + "\\" + files[img] + str(i) + "-" + str(j) + ".tif"

            out_ds = tif_driver.Create(file,b_xsize,b_ysize,1,gdal.GDT_Byte)

            ori_transform = in_ds.GetGeoTransform()

            top_left_x = ori_transform[0]
            w_e_pixel_resolution = ori_transform[1]
            top_left_y = ori_transform[3]
            n_s_pixel_resolution = ori_transform[5]

            top_left_x = top_left_x + offset_x * w_e_pixel_resolution
            top_left_y = top_left_y + offset_y * n_s_pixel_resolution

            dst_transform = (top_left_x,ori_transform[1],ori_transform[2],top_left_y,ori_transform[4],ori_transform[5])

            out_ds.SetGeoTransform(dst_transform)
            out_ds.SetProjection(im_proj)
            out_ds.GetRasterBand(1).WriteArray(im_data_normalized)
            out_ds.FlushCache()
            del out_ds
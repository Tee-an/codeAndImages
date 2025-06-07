import os.path

import rasterio
from osgeo import gdal
from rasterio.transform import Affine
from rasterio.merge import merge
from tqdm import tqdm

import numpy as np
from PIL import Image

def png_to_geotiff(png_path,ref_tif_path,output_geotiff_path):
    with rasterio.open(ref_tif_path) as src:
        transform = src.transform
        crs = src.crs
        width,height = src.width,src.height

    png_image = Image.open(png_path).convert("L")
    mask_array = np.array(png_image)

    if mask_array.shape != (height,width):
        raise ValueError("尺寸不匹配！")

    with rasterio.open(
        output_geotiff_path,
        "w",
        driver="GTiff",
        height = height,
        width = width,
        count = 1,
        dtype = mask_array.dtype,
        crs = crs,
        transform = transform,
        nodata = 0
    )as dst:
        dst.write(mask_array,1)

def merge_tifs(tmp_tifDir,geotiff_paths,output_path):
    #src_files = [rasterio.open(tmp_tifDir + "/" + f) for f in geotiff_paths]
    src_files = []
    print(geotiff_paths)
    for f in geotiff_paths:
        str = tmp_tifDir + "/" + f
        print(str)
        src_files.append(rasterio.open(str,"r+"))

    print(src_files)
    mosaic,transform = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform
    })

    # 写入拼接结果
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # 关闭所有文件
    for src in src_files:
        src.close()

    print(f"拼接完成: {output_path}")

tifDir = "../data/output3"
pngDir = "../data/prediction3"
tmp_geotiffDir = "../data/merge/tmp_tif/"
if not os.path.exists(tmp_geotiffDir):
    os.mkdir(tmp_geotiffDir)
output_merged = "../data/merge/merged-1520.tif"

original_tifs = [i for i in os.listdir(tifDir)]
predict_png = [i for i in os.listdir(pngDir)]

tmp_tifDir = "../data/merge/tmp_tif"
geotiff_paths = [i for i in os.listdir(tmp_tifDir)]
print(len(geotiff_paths))
for tif_path, png_path in zip(original_tifs, predict_png):
    tmp_trans_path = os.path.basename(png_path).replace(".png", ".tif")
    output_geotiff = os.path.join(tmp_geotiffDir, tmp_trans_path)
    print(output_geotiff)
    png_to_geotiff(pngDir + "/" + png_path, tifDir + "/" + tif_path, output_geotiff)
    print(len(geotiff_paths))
    geotiff_paths.append(tmp_trans_path)


merge_tifs(tmp_tifDir,geotiff_paths,output_merged)
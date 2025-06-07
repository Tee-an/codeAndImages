import os
import shutil

fileDir = "../data/ground"
maskDir = "../data/prediction"
print(os.listdir(fileDir))

files = []
for i in os.listdir(fileDir):
    sample_Dir = fileDir + '/' + i
    print(sample_Dir)
    for j in os.listdir(sample_Dir):
        old_path = maskDir + '/' + j.split('.tif')[0] + "-mask.png"
        print(old_path)
        new_path = sample_Dir + '/' + j.split('.tif')[0] + "-mask.png"
        print(new_path)
        shutil.copy(old_path,new_path)


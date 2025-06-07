import os

fileDir = "../data/train_data"
files = [i.split(".tif")[0] for i in os.listdir(fileDir) if i.endswith(".tif")]
masks = [i.split(".json")[0] for i in os.listdir(fileDir) if i.endswith(".json")]
print(files)
print(masks)

for i in files:
    if i not in masks:
        print(i)
        os.remove(fileDir + "/" + i + ".tif")
import cv2
file = cv2.imread("../data/ground/1/ESP_020782_1610_RED.JP231-24-mask.png",cv2.IMREAD_GRAYSCALE)
file2 = cv2.imread("../data/ground/1-ground/ESP_020782_1610_RED.JP231-24.json-mask.png",cv2.IMREAD_GRAYSCALE)

h,w = file.shape
print(h,w)
D = (int)(h * 0.05)
num1 = 0
num2 = 0
diffNum = 0
sameNum = 0
for i in range(h):
    for j in range(w):
        if file[i][j] > 0:
            num1 += 1
        else:
            num2 += 1

print(num1,num2)

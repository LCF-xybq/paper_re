from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imageio

pth = r'D:\Program_self\paper_re\CURL\result\0.png'

img = Image.open(pth).convert("RGB")
img2 = cv2.imread(pth)



plt.imshow(img2)
plt.show()
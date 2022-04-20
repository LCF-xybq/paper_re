from PIL import Image
import matplotlib.pyplot as plt

pth = r'D:\Program_self\paper_re\data\UW\train\ref\2_img_.png'

img = Image.open(pth)

plt.imshow(img)
plt.show()
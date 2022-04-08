from PIL import Image
import torch

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)

    return img

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def gram(y):
    # [ch h w] ==> [ch h*w], 记为A
    # gram = A*A^T / c*h*w
    # y is tensor，[batch ch h w]

    (b, c, h, w) = y.size()
    A = y.view(b, c, h * w)
    A_T = A.transpose(1, 2)
    matrix = A.bmm(A_T) / (c * h * w)
    return matrix
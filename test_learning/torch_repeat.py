import torch

if __name__ == '__main__':
    img = torch.randint(1, 9, (2, 3, 2, 2)) # batch channel h w
    print(img)

    to_test_img = torch.randint(1, 9, (3, 2, 2))
    print('-' * 30)
    print(to_test_img)
    to_test_img_repeat = to_test_img.repeat(2, 1, 1, 1)
    print('-' * 30)
    print(to_test_img_repeat)

    # 复制batchsize个图片
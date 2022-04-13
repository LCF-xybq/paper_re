import os

hr_pth = r'D:\Program_self\paper_re\data\SR\train\hr'
lr_pth = r'D:\Program_self\paper_re\data\SR\train\lr'
ann = r'D:\Program_self\paper_re\data\SR\ann.txt'

if __name__ == '__main__':
    assert os.path.isdir(hr_pth)
    assert os.path.isdir(lr_pth)

    img_list = os.listdir(hr_pth)
    with open(ann, 'a') as f:
        for hr_name in img_list:
            basename, ext = os.path.splitext(hr_name)
            hr_img_pth = os.path.join(hr_pth, hr_name)
            lr_name = basename + 'x4' + ext
            lr_img_pth = os.path.join(lr_pth, lr_name)
            f.write(f'{lr_img_pth} {hr_img_pth}\n')
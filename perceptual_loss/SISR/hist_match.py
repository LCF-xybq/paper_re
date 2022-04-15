import cv2
import numpy as np
import matplotlib.pyplot as plt


def plt_hist(img):
    blue_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    green_hist = cv2.calcHist([img], [1], None, [256], [0, 255])
    red_hist = cv2.calcHist([img], [2], None, [256], [0, 255])

    plt.plot(blue_hist, color='blue')
    plt.plot(green_hist, color='green')
    plt.plot(red_hist, color='red')
    plt.show()

def hist_match(src , ref):
    src_shape = src.shape

    src_ravel = src.ravel()
    ref_ravel = ref.ravel()

    o_values, bin_idx, o_counts = np.unique(src_ravel, return_inverse=True, return_counts=True)
    b_values, b_counts = np.unique(ref_ravel, return_counts=True)

    o_quantiles = np.cumsum(o_counts).astype(np.float64)
    o_quantiles /= o_quantiles[-1]
    b_quantiles = np.cumsum(b_counts).astype(np.float64)
    b_quantiles /= b_quantiles[-1]

    interp_t_values = np.interp(o_quantiles, b_quantiles, b_values)

    return interp_t_values[bin_idx].reshape(src_shape)

if __name__ == '__main__':
    lr_pth = r'D:\Program_self\paper_re\data\SR\test\lr\img_005_SRF_4_LR.png'
    output_pth = r'D:\Program_self\paper_re\perceptual_loss\results\sisr\resultx4.jpg'

    img_raw = cv2.imread(output_pth)
    img_ref = cv2.imread(lr_pth)

    plt_hist(img_ref)

    result = np.copy(img_raw)
    for i in range(3):
        result[:, :, i] = hist_match(img_raw[:, :, i], img_ref[:, :, i])

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.show()

    plt_hist(result)




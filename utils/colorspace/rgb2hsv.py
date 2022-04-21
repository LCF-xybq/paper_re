import torch
import torch.nn as nn

"""
ToTensor img: [0.0, 1.0]

0<=H<=360
0<=S<=1
0<=V<=1

R' = R/255
G' = G/255
B' = B/255
cmax = max(R', G', B')
cmin = min(R', G', B')
delta = cmax - cmin

H = 0, delta = 0
H = 60/360 * ((G' - B')/delta + 0), cmax=R'
H = 60/360 * ((B' - R')/delta + 2), cmax=G'
H = 60/360 * ((R' - G')/delta + 4), cmax=B'

S = 0, cmax = 0 # this is meaningless
S = delta/cmax, cmax != 0

V = cmax

if H < 0:
    H = H + 360

finish

then convert to [0, 1]
H %= 360
H /= 360

"""

class RGB2HSV(nn.Module):
    def __init__(self):
        super(RGB2HSV, self).__init__()

    def forward(self, img):
        batch, ch, h, w = img.shape
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        cmax, cmax_id = torch.max(img, dim=1)
        cmin, _ = torch.min(img, dim=1)
        delta = cmax - cmin
        # avoid division by 0
        S = delta / (cmax + 0.0001)
        H = torch.zeros_like(img[:, 0, :, :])
        # cmax == R'
        mark = cmax_id == 0
        H[mark] = 60 * (g[mark] - b[mark]) / (delta[mark] + 0.0001)
        # cmax == G'
        mark = cmax_id == 1
        H[mark] = 120 + 60 * (b[mark] - r[mark]) / (delta[mark] + 0.0001)
        # cmax == B'
        mark = cmax_id == 2
        H[mark] = 240 + 60 * (r[mark] - g[mark]) / (delta[mark] + 0.0001)

        mark = H < 0
        H[mark] += 360
        H = H % 360
        H = H / 360

        result = torch.cat([H.view(batch, 1, h, w), S.view(batch, 1, h, w), cmax.view(batch, 1, h, w)], 1)

        return result

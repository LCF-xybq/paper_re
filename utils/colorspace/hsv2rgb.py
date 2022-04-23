import torch
import torch.nn as nn

"""

R = V - ((S*V)/60)delta(360h - 60) + ((S*V)/60)delta(360h - 240)
G = V(1 - S) - ((S*V)/60)delta(360h - 60) + ((S*V)/60)delta(360h - 180)
B = V(1 - S) - ((S*V)/60)delta(360h - 120) + ((S*V)/60)delta(360h - 300)

"""
class HSV2RGB(nn.Module):
    def __init__(self):
        super(HSV2RGB, self).__init__()

    def forward(self, img):
        batch, ch, height, w = img.shape
        h, s, v = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        htemp = (h * 360) % 360
        h = htemp / 360

        vs = torch.div(torch.mul(v, s), 60)
        # delta(360h - 60)
        r1_delta = delta(torch.add(torch.mul(h, 360), -60))
        # delta(360h - 240)
        r2_delta = delta(torch.add(torch.mul(h, 360), -240))

        # delta(360h - 0)
        g1_delta = delta(torch.add(torch.mul(h, 360), 0))
        # delta(360h - 180)
        g2_delta = delta(torch.add(torch.mul(h, 360), -180))

        # delta(360h - 120)
        b1_delta = delta(torch.add(torch.mul(h, 360), -120))
        # delta(360h - 300)
        b2_delta = delta(torch.add(torch.mul(h, 360), -300))

        # (1 - S)
        one_minus_s = torch.mul(torch.add(s, -1), -1)

        # R = V - ((S*V)/60)delta(360h - 60) + ((S*V)/60)delta(360h - 240)
        r1 = torch.mul(torch.mul(vs, r1_delta), -1)
        r2 = torch.mul(vs, r2_delta)
        R = torch.add(torch.add(v, r1), r2)

        # G = V(1 - S) + ((S*V)/60)delta(360h - 0) - ((S*V)/60)delta(360h - 180)
        g1 = torch.add(torch.mul(v, one_minus_s), torch.mul(vs, g1_delta))
        g2 = torch.mul(vs, g2_delta)
        G = torch.add(torch.mul(g2, -1), g1)

        # B = V(1 - S) + ((S*V)/60)delta(360h - 120) - ((S*V)/60)delta(360h - 300)
        b1 = torch.add(torch.mul(v, one_minus_s), torch.mul(vs, b1_delta))
        b2 = torch.mul(vs, b2_delta)
        B = torch.add(torch.mul(b1, -1), b2)

        del h, s, vs, r1_delta, r2_delta, g1_delta, g2_delta, b1_delta, b2_delta, one_minus_s, r1, r2, g1, g2, b1, b2

        R = torch.reshape(R, (batch, 1, height, w))
        G = torch.reshape(G, (batch, 1, height, w))
        B = torch.reshape(B, (batch, 1, height, w))

        RGB = torch.cat([R, G, B], dim=1)

        return RGB

def delta(input):
    out = torch.clamp(input, min=0, max=60)
    return out
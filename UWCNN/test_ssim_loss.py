from loss.ssim_loss import SSIMLoss
from uw_dataset import UWCNNData
from torchvision import transforms
from torch.utils.data import DataLoader
from uwcnn_model import UWCNN


if __name__ == "__main__":
    ann_file = r'D:\Program_self\paper_re\data\UW\ann.txt'

    transform = transforms.Compose([
        transforms.Resize(550),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    test_dataset = UWCNNData(ann_file, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UWCNN().cuda()
    ssim = SSIMLoss()
    for i, data in enumerate(test_loader):
        raw = data['raw'].cuda()
        ref = data['ref'].cuda()

        out = model(raw)
        loss = ssim(out, ref)
        loss_metric = loss.cpu().detach().numpy()
        print(loss, loss_metric)
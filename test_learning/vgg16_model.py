from torchvision import models

if __name__ == '__main__':
    model = models.vgg16()
    vgg_pretrained_features = models.vgg16(pretrained=True).features
    print(vgg_pretrained_features)
from uwcnn_model import UWCNN


if __name__ == '__main__':
    model1 = UWCNN()
    print(model1)
    num1 = [p.numel() for p in model1.parameters()]
    print(model1.state_dict()['block1.conv1.weight'])
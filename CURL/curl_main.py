import json
import os
import sys

class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

def getargs():
    with open('curl_setting.json', 'r') as f:
        data = json.load(f, object_hook=JSONObject)

    return data

def check_paths(args):
    try:
        if args.save_pth is not None and not (os.path.exists(args.save_pth)):
            os.makedirs(args.save_pth)
        if args.ckpt is not None and not (os.path.exists(args.ckpt)):
            os.makedirs(args.ckpt)
        if args.work_dir is not None and not (os.path.exists(args.work_dir)):
            os.makedirs(args.work_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def train(args):
    pass

def test(args):
    pass

def main():
    args = getargs()

    if args.train:
        check_paths(args)
        train(args)
    else:
        test(args)

if __name__ == '__main__':
    main()
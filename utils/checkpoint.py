import os
import torch
from collections import OrderedDict

def load(filename, model, logger):
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        checkpoint = torch.load(filename,
                                map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        checkpoint = torch.load(filename)
    if "state_dict" in checkpoint.keys():
        checkpoint = remove_prefix(checkpoint['state_dict'], 'module.')
    else:
        checkpoint = remove_prefix(checkpoint, 'module.')
    model.load_state_dict(checkpoint)
    if logger is not None:
        logger.info('load checkpoint from %s', filename)

def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def resume(filename, model, optimizer, logger, resume_optimizer=True,):
    assert isinstance(filename, str)
    assert os.path.exists(filename)
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        checkpoint = torch.load(filename,
                                map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        checkpoint = torch.load(filename)
    if "state_dict" in checkpoint.keys():
        checkpoint = remove_prefix(checkpoint['state_dict'], 'module.')
    else:
        checkpoint = remove_prefix(checkpoint, 'module.')
    model.load_state_dict(checkpoint)
    logger.info('load checkpoint from %s', filename)
    epoch = checkpoint['meta']['epoch']
    iter = checkpoint['meta']['iter']
    if 'optimizer' in checkpoint and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info('resumed epoch %d, iter %d', epoch, iter)
    return epoch, iter

def save_latest(model, optimizer, out_dir, epoch, iters, save_optimizer=True, meta=None):
    if meta is None:
        meta = dict(epoch=epoch + 1, iter=iters)
    elif isinstance(meta, dict):
        meta.update(epoch=epoch + 1, iter=iters)
    else:
        raise TypeError(
            f'meta should be a dict or None, but got {type(meta)}')
    if save_optimizer:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict()),
            'optimizer': optimizer.state_dict()}
    else:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict())}
    save_path = os.path.join(out_dir, 'latest.pth')
    torch.save(checkpoint, save_path)

def save_epoch(model, optimizer, out_dir, epoch, iters, save_optimizer=True, meta=None):
    if meta is None:
        meta = dict(epoch=epoch + 1, iter=iters)
    elif isinstance(meta, dict):
        meta.update(epoch=epoch + 1, iter=iters)
    else:
        raise TypeError(
            f'meta should be a dict or None, but got {type(meta)}')
    if save_optimizer:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict()),
            'optimizer': optimizer.state_dict()}
    else:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict())}

    save_path = out_dir + '/epoch_{}.pth'.format(epoch + 1)
    torch.save(checkpoint, save_path)

def weights_to_cpu(state_dict):
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu

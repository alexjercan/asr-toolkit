import os
import torch
import torch.nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
            type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def set_parameter_requires_grad(model, val=False):
    for param in model.parameters():
        param.requires_grad = val


def tensors_to_device(tensors, device):
    return (tensor.to(device, non_blocking=True) if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors)


def load_checkpoint(model, checkpoint_file, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def save_checkpoint(model, dir_checkpoints, file_name):
    output_path = os.path.join(dir_checkpoints, file_name)
    if not os.path.exists(dir_checkpoints):
        os.makedirs(dir_checkpoints)
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, output_path)


def tensor_to_string(tensor, target_lengths, labels):
    result = []
    s = 0
    for i in range(target_lengths.shape[0]):
        result.append("".join([labels[t.item()] for t in tensor[s:s+target_lengths[i]]]))
        s = s + target_lengths[i]
    return result


def pad_last(tensor):
    maxlen = max(map(lambda t: t.shape[-1], tensor))
    return list(map(lambda t: F.pad(t, (0, maxlen - t.shape[-1])), tensor))

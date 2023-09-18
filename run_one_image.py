import sys
import os
import requests

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import models_vit

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def prepare_model(chkpt_dir, arch='vit_base_patch16'):
    # build model
    # model = getattr(models_vit, arch)()
    # # load model
    # checkpoint = torch.load(chkpt_dir, map_location='cpu')
    # msg = model.load_state_dict(checkpoint['model'], strict=False)
    # print(msg)
    model = models_vit.__dict__[arch](
        num_classes=1000,
        drop_path_rate=0.1,
        global_pool=True,
    )
    class Identity(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return x
    model.head = Identity()
    model.fc_norm = Identity()
    print(model)
    # model = torch.nn.Sequential(*(list(model.children())[:-2]))
    print("new model")
    print(model)
    # exit(0)
    if chkpt_dir:
        checkpoint = torch.load(chkpt_dir, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % chkpt_dir)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        global_pool = True
        # if global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
    return model

img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((224, 224))
img = np.array(img) / 255.

assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

chkpt_dir = '/raid/varsha/MAE/pre_trained_ckpt/mae_pretrain_vit_base.pth'
# chkpt_dir = '/raid/varsha/MAE/fine_tuned_ckpt/mae_finetuned_vit_base.pth'
model_mae = prepare_model(chkpt_dir, 'vit_base_patch16')
# print(model_mae)
# print('Model loaded.')

x = torch.tensor(img)

# make it a batch-like
x = x.unsqueeze(dim=0)
x = torch.einsum('nhwc->nchw', x)

loss = model_mae(x.float())
print(loss.shape)
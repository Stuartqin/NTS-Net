"""编写一个方法执行前向传播"""
import os
import sys
from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
sys.path.append('.')
from config import BATCH_SIZE, PROPOSAL_NUM, test_model, INPUT_SIZE
from core import model, dataset
# from core.utils import progress_bar
import scipy.misc
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='visual proposal region over input image!')
    parser.add_argument('--image', dest='image', type=str)
    args = parser.parse_args()
    image_path = args.image

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not test_model:
        raise NameError('please set the test_model file to choose the checkpoint!')
    # read dataset
    # trainset = dataset.CUB(root='./CUB_200_2011', is_train=True, data_len=None)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
    #                                           shuffle=True, num_workers=8, drop_last=False)
    # testset = dataset.CUB(root='./CUB_200_2011', is_train=False, data_len=None)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
    #                                          shuffle=False, num_workers=8, drop_last=False)
    # define model

    net = model.attention_net(topN=PROPOSAL_NUM)
    ckpt = torch.load(test_model)
    net.load_state_dict(ckpt['net_state_dict'])
    net = net.cuda()
    net.eval()

    # 截取选出的三个anchor区域
    # img = scipy.misc.imread('C:/Users/14014/Desktop/2012/1532079420425.jpg')
    img = scipy.misc.imread(image_path)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')

    img = Image.fromarray(img, mode='RGB')
    img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = torch.from_numpy(np.expand_dims(img.numpy(), axis=0))
    print(img.shape)
    with torch.no_grad():
        _, concat_logits, _, _, _ = net(img.cuda())
        part_imgs = net.part_imgs
        print(len(part_imgs))
        # print(len(concat_logits))
    # plot显示出来
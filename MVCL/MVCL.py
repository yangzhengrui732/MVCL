# -*- coding: utf-8 -*-

import datetime
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader, TensorDataset

import record
from Model import mae_vit_HSI_patch3, vit_HSI_patch3, mae_vit_HSI_patch3_1
from augment import CenterResizeCrop, random_crop, extract_view_2
from config import load_args
from data_read import readdata, load_data_true
from hyper_dataset import HyperData
from nt_xent import NTXentLoss
from pos_embed import interpolate_pos_embed
from util_CNN import test_batch, pre_train
import matplotlib.pyplot as plt
import torch.utils.data as Data
from matplotlib import colors
from scipy.io import loadmat


args = load_args()
torch.cuda.set_device(1)

mask_ratio = args.mask_ratio
windowsize = args.windowsize
dataset = args.dataset
type = args.type
num_epoch = args.epochs
lr = args.lr
train_num_per = args.train_num_perclass
num_of_ex = 1  # Run it a total of how many times and then take the average.
batch_size = args.batch_size
net_name = args.cl_mode + '_MVCL'

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')
halfsize = int((windowsize - 1) / 2)
_, _, _, _, _, _, _, _, _, _, gt, s = readdata(type, dataset, windowsize, train_num_per, 1000, 0)
num_of_samples = int(s * 0.2)
nclass = np.max(gt)
print(nclass)

def gen_ran_output(data, label, model, vice_model, args):
    for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            # adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)
            adv_param.data = param.data + args.eta * torch.normal(0, torch.ones_like(param.data) * param.data.std() * 1)
    z2 = vice_model(data, label, mask_ratio=mask_ratio, mode=args.cl_mode, temp=args.cl_temperature)
    return z2

# -------------------------------------------------------------------------------
# Locate the training and testing samples.
def chooose_train_and_test_point(true_data, num_classes):
    number_true = []
    pos_true = {}
    # --------------------------for true data------------------------------------
    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_true, number_true

# Labels y_train, y_test
def train_and_test_label(number_true, num_classes):
    y_true = []
    for i in range(num_classes + 1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_true = np.array(y_true)
    return  y_true

# Obtain the image data of the patch
def gain_neighborhood_pixel(mirror_image, point, i, patch=27):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

# Summarize all the data
def true_data(mirror_image, band, true_point, patch=5):
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=np.float32)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_true.dtype))
    print("**************************************************")
    return x_true

def test_epoch(model, test_loader):
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((num_of_ex, nclass))
af_result = np.zeros([nclass + 3, num_of_ex])
criterion = nn.CrossEntropyLoss()
crop_size = 9


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def forward_contrastive(x1, x2, y=None, mode='Siam', temperature=1.0, global_pool=False):
    # ph1 = Block(args.encoder_dim, 8, args.mlp_ratio, qkv_bias=True).cuda()
    ph1 = Block(args.encoder_dim, 8, 2.0, qkv_bias=True).cuda()
    # hidden_dim = 64
    # projection_mlp = nn.Sequential(
    #     nn.Linear(args.encoder_dim, hidden_dim),
    #     nn.BatchNorm1d(hidden_dim),
    #     # nn.LeakyReLU(0.2, inplace=True),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(hidden_dim, args.encoder_dim)
    # )
    #
    # h = projection_mlp
    if global_pool:
        z1 = x1[:, 1:, :].mean(dim=1)  # global pool without cls token
        z2 = x2[:, 1:, :].mean(dim=1)
    else:
        z1 = x1[:, 0, :]  # with cls token
        z2 = x2[:, 0, :]
    if mode == 'SimCLR':
        z1 = ph1(x1)[:, 0, :]
        z2 = ph1(x2)[:, 0, :]
        cl_loss = NTXentLoss(device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                             temperature=temperature, batch_size=x1.size(0))
        L = cl_loss(z1, z2)
    else:
        raise Exception('loss type must be selected from {SimCLR, Siam, SimCLR_A}')
    return L


for num in range(0, num_of_ex):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('num:', num)
    train_image, train_label, validation_image1, validation_label1, nTrain_perClass, nvalid_perClass, train_index, val_index, index, image, gt, s = readdata(
        type, dataset, windowsize, train_num_per, num_of_samples, num)
    ind = np.random.choice(validation_image1.shape[0], 100, replace=False)

    validation_image = validation_image1[ind]
    validation_label = validation_label1[ind]
    nvalid_perClass = np.zeros_like(nvalid_perClass)
    nband = train_image.shape[3]

    train_num = train_image.shape[0]  # 180
    train_image = np.transpose(train_image, (0, 3, 1, 2))

    validation_image = np.transpose(validation_image, (0, 3, 1, 2))
    validation_image1 = np.transpose(validation_image1, (0, 3, 1, 2))

    if args.augment:
        transform_train = [CenterResizeCrop(scale_begin=args.scale, windowsize=windowsize)]
        train_dataset = HyperData((train_image, train_label), transform_train)
    else:
        train_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_label))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print("=> creating model '{}'".format(net_name))

    net = mae_vit_HSI_patch3(img_size=(crop_size, crop_size), in_chans=nband, hid_chans=args.hid_chans,
                             embed_dim=args.encoder_dim,
                             depth=args.encoder_depth, num_heads=args.encoder_num_heads, mlp_ratio=args.mlp_ratio,
                             nb_classes=nclass)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    net1 = mae_vit_HSI_patch3_1(img_size=(crop_size, crop_size), in_chans=nband, hid_chans=args.hid_chans,
                                embed_dim=args.encoder_dim,
                                depth=args.encoder_depth, num_heads=args.encoder_num_heads, mlp_ratio=args.mlp_ratio,
                                nb_classes=nclass)
    net1.cuda()
    optimizer1 = optim.Adam(net1.parameters(), lr=lr, weight_decay=1e-4)

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=5, T_mult=2)

    tic1 = time.time()
    for epoch in range(num_epoch):
        net.train()
        net1.train()
        total_loss = 0
        for idx, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            # _, cl_loss, _, _, _ = net(x, y,mask_ratio=mask_ratio, mode = args.cl_mode, temp = args.cl_temperature)
            # _, cl_loss, _, _, _ = net1(x, y,mask_ratio=mask_ratio, mode = args.cl_mode, temp = args.cl_temperature)

            # -------------------------------------------------------------------------------
            # view1 = extract_view_2(x,crop_size)
            # view1 = view1.cuda()
            #
            # view1_ori_dim = view1.size(1)
            # view1 = view1.view(view1.size(0), view1_ori_dim, -1)
            # view1 = view1.permute(0, 2, 1)
            # view1 = view1.view(view1.size(0), view1.size(1), int(view1_ori_dim ** 0.5), int(view1_ori_dim ** 0.5))
            # conv = nn.Conv2d(view1.size(1), x.size(1), kernel_size=1, stride=1, padding=0,
            #                  device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
            # view1 = conv(view1)
            # -------------------------------------------------------------------------------
            view1 = random_crop(x, crop_size)
            view1 = view1.cuda()

            view1_feature, y, mode, temp, logits_1 = net(view1, y, mask_ratio=mask_ratio, mode=args.cl_mode,
                                                         temp=args.cl_temperature)
            # view1_feature, y, mode, temp, _ = net(view1, y, mask_ratio=mask_ratio, mode=args.cl_mode, temp=args.cl_temperature)

            view2 = extract_view_2(x, crop_size)
            view2 = view2.cuda()

            view2_ori_dim = view2.size(1)
            view2 = view2.view(view2.size(0), view2_ori_dim, -1)
            view2 = view2.permute(0, 2, 1)
            view2 = view2.view(view2.size(0), view2.size(1), int(view2_ori_dim ** 0.5), int(view2_ori_dim ** 0.5))
            conv = nn.Conv2d(view2.size(1), x.size(1), kernel_size=1, stride=1, padding=0,
                             device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
            view2 = conv(view2)

            view2_feature, _, _, _, logits_2 = net1(view2, y, mask_ratio=mask_ratio, mode=args.cl_mode,
                                                    temp=args.cl_temperature)

            cls_loss_1 = criterion(logits_1 / args.temperature, y) * args.cls_loss_ratio
            cls_loss_2 = criterion(logits_2 / args.temperature, y) * args.cls_loss_ratio1

            cl_loss = forward_contrastive(view1_feature, view2_feature, y, mode, temp)

            loss = args.cl_loss_ratio * cl_loss + cls_loss_1 + cls_loss_2
            optimizer.zero_grad()
            optimizer1.zero_grad()

            loss.backward()
            # print("Model Parameters:")
            # for name, param in net1.named_parameters():
            #     print(
            #         f"Parameter name: {name}, Size: {param.size()}, requires_grad: {param.requires_grad}, Values: {param.data}")

            optimizer.step()
            optimizer1.step()
            total_loss = total_loss + loss

        scheduler.step()
        scheduler1.step()
        total_loss = total_loss / (idx + 1)
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        state1 = {'model': net1.state_dict(), 'optimizer': optimizer1.state_dict(), 'epoch1': epoch}

        print('epoch:', epoch, 'loss:', total_loss.data.cpu().numpy())
        # print('epoch1:', epoch, 'loss1:', total_loss.data.cpu().numpy())

    toc1 = time.time()
    torch.save(state, './net.pt')
    torch.save(state1, './net1.pt')

    # ########################   vit的finetune 
    model = vit_HSI_patch3(img_size=(windowsize, windowsize), in_chans=nband, hid_chans=args.hid_chans,
                           embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,
                           mlp_ratio=args.mlp_ratio, num_classes=nclass, global_pool=False).cuda()
    checkpoint = torch.load('./net.pt')
    checkpoint_model = checkpoint['model']

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)  # 通过插值的方式,将预训练模型中的position embedding缩放到新的模型大小,以便加载预训练的参数
    msg = model.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)  # 初始化函数，避免初始的权重过大
    # trunc_normal_(model.head.weight, std=1e-6)  # 初始化函数，避免初始的权重过大

    # torch.nn.init.xavier_uniform_(model.head.weight)
    tic2 = time.time()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 140], gamma=0.1, last_epoch=-1)

    model = pre_train(model, train_image, train_label, validation_image, validation_label, 180, optimizer, scheduler,
                      batch_size, val=False)
    toc2 = time.time()

    true_cla, overall_accuracy, average_accuracy, kappa, true_label, test_pred, test_index, cm, pred_array = test_batch(
        model.eval(), image, index, 100, nTrain_perClass, nvalid_perClass, halfsize)
    toc3 = time.time()

    af_result[:nclass, num] = true_cla
    af_result[nclass, num] = overall_accuracy
    af_result[nclass + 1, num] = average_accuracy
    af_result[nclass + 2, num] = kappa

    OA.append(overall_accuracy)
    AA.append(average_accuracy)
    KAPPA.append(kappa)
    TRAINING_TIME.append(toc1 - tic1 + toc2 - tic2)
    TESTING_TIME.append(toc3 - toc2)
    ELEMENT_ACC[num, :] = true_cla
    # classification_map, gt_map = generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred,
    #                                       overall_accuracy, halfsize, dataset, day_str, num, net_name)
    # # output classification maps
    # color_mat = loadmat('./datasets/KSC/KSC_colormap_new.mat')
    # color_mat = loadmat('./datasets/Indian/Indian_colormap.mat')
    color_mat = loadmat('./datasets/Houston2013/HU2013_colormap.mat')
    color_mat_list = list(color_mat)
    color_matrix1 = color_mat[color_mat_list[3]]  # (17,3)
    color_matrix = color_matrix1[1:, :]

    num_classes = np.max(gt)


    # image_gr, label_gr = load_data_true('Indian')
    # image_gr, label_gr = load_data_true('KSC')
    image_gr, label_gr = load_data_true('Houston2013')
    height = image_gr.shape[0]
    width = image_gr.shape[1]
    band = image_gr.shape[2]
    total_pos_true, number_true = chooose_train_and_test_point(label_gr, num_classes)

    part = 10000

    number = total_pos_true.shape[0] // part
    # pre_u = np.empty(number, dtype='float32')
    pre_u = np.empty(total_pos_true.shape[0], dtype='float32')
    for i in range(number):
        x_true_band = true_data(image, band, total_pos_true[i * part: (i + 1) * part, :], patch=11)

        x_true = torch.from_numpy(x_true_band.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        y_true = train_and_test_label(number_true, num_classes)
        y_true1 = y_true[i * part: (i + 1) * part]
        y_true1 = torch.from_numpy(y_true1).type(torch.LongTensor)
        Label_true = Data.TensorDataset(x_true, y_true1)
        label_true_loader = Data.DataLoader(Label_true, batch_size=64, shuffle=False)

        pre_u[i * part: (i + 1) * part] = test_epoch(model, label_true_loader)

        # pre_u[i * part: (i + 1) * part] = model(torch.tensor(x_true).cuda())

    if (i + 1) * part < total_pos_true.shape[0]:
        count = 0
        x_true_band = true_data(image, band, total_pos_true[(i + 1) * part: total_pos_true.shape[0], :], patch=11)

        x_true = torch.from_numpy(x_true_band.transpose(0, 3, 1, 2)).type(torch.FloatTensor)

        y_true1 = y_true[(i + 1) * part: total_pos_true.shape[0]]
        y_true1 = torch.from_numpy(y_true1).type(torch.LongTensor)
        Label_true = Data.TensorDataset(x_true, y_true1)
        label_true_loader = Data.DataLoader(Label_true, batch_size=64, shuffle=False)
        pre_u[(i + 1) * part: total_pos_true.shape[0]] = test_epoch(model, label_true_loader)
        # pre_u[(i + 1) * part: total_pos_true.shape[0]] = model(torch.tensor(x_true).cuda())

    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1
    plt.subplot(1, 1, 1)
    plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.savefig('./datasets/Indian_CL.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.savefig('./datasets/Houston2013.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
    # plt.savefig('./datasets/Houston_CL.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
jingdu = np.mean(af_result, axis=1)
print("--------" + net_name + " Training Finished-----------")
# 假设模型为 net
out_parameters = count_parameters(net)
out_parameters1 = count_parameters(net1)
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     './records/' + dataset + '/' + net_name + '_' + day_str + '_' + str(
                         args.epochs) + '_train_num：' + str(train_image.shape[0]) + '_windowsize：' + str(
                         windowsize) + '_lr_' + str(args.lr) + '_mask_ratio_' + str(mask_ratio) + '_temperature_' + str(
                         args.temperature) +
                     '_augment_' + str(args.augment) + '_aug_scale_' + str(args.scale) + '_CLloss_ratio_' + str(
                         args.cl_loss_ratio) + '_cl_temp_' + str(args.cl_temperature) + '_loss_ratio_' + str(
                         args.cls_loss_ratio)+ str(args.cls_loss_ratio1) + '.txt')

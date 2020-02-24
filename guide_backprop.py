import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function

from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D

np.random.seed(1000)


def r2plus1d_34(num_classes, pretrained=False, progress=False, **kwargs):
    model = VideoResNet(block=BasicBlock,
                        conv_makers=[Conv2Plus1D] * 4,
                        layers=[3, 4, 6, 3],
                        stem=R2Plus1dStem)

    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    return model


def show_frames_on_figure(frames):
    fig = plt.figure(figsize=(10, 10))
    rows = 4
    columns = 8
    for i, frame in enumerate(frames):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(frame)
    plt.show()


def get_input_frames(args):
    cap = cv2.VideoCapture(args.video_path)

    frame_list = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (112, 112))
        frame_list.append(frame.copy()[:, :, ::-1])

    choice_list = range(len(frame_list) - 32 + 1)
    index = np.random.choice(list(choice_list))
    print("frame index:", index)
    frame_list = frame_list[index:index + 32]
    frames_to_show = frame_list.copy()
    show_frames_on_figure(frames_to_show)
    frame_list = np.stack(frame_list, axis=0)
    frame_list = np.transpose(frame_list, axes=(3, 0, 1, 2))

    frame_list = torch.from_numpy(frame_list)
    frame_list = frame_list.unsqueeze(0).float().requires_grad_(True)

    return frame_list


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.cuda = use_cuda

        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model._modules.items():
            # print(idx, module)
            if len(module._modules.keys()) == 0:
                print("idx, module", idx, module)
                if module.__class__.__name__ == 'ReLU':
                    print("Add module to GuidedBackpropReLU")
                    print(self.model._modules[idx])
                    self.model._modules[idx] = GuidedBackpropReLU.apply
            else:
                for subidx, submodule in module._modules.items():
                    if len(submodule._modules.keys()) == 0:
                        print("subidx, submodule", subidx, submodule)
                        if submodule.__class__.__name__ == 'ReLU':
                            print("Add module to GuidedBackpropReLU")
                            print(self.model._modules[idx]._modules[subidx])
                            self.model._modules[idx]._modules[subidx] = GuidedBackpropReLU.apply
                    else:
                        for sub1idx, sub1module in submodule._modules.items():
                            if len(sub1module._modules.keys()) == 0:
                                print("sub1idx, sub1module", sub1idx, sub1module)
                                if sub1module.__class__.__name__ == 'ReLU':
                                    print("Add module to GuidedBackpropReLU")
                                    print(self.model._modules[idx]._modules[subidx]._modules[sub1idx])
                                    self.model._modules[idx]._modules[subidx]._modules[
                                        sub1idx] = GuidedBackpropReLU.apply
                            else:
                                for sub2idx, sub2module in sub1module._modules.items():
                                    if len(sub2module._modules.keys()) == 0:
                                        print("sub2idx, sub2module", sub2idx, sub2module)
                                        if sub2module.__class__.__name__ == 'ReLU':
                                            print("Add module to GuidedBackpropReLU")
                                            print(self.model._modules[idx]._modules[subidx]._modules[
                                                      sub1idx]._modules[sub2idx])
                                            self.model._modules[idx]._modules[subidx]._modules[sub1idx]._modules[
                                                sub2idx] = GuidedBackpropReLU.apply
                                    else:
                                        for sub3idx, sub3module in sub2module._modules.items():
                                            print("sub3idx, sub3module", sub3idx, sub3module)
                                            if sub3module.__class__.__name__ == 'ReLU':
                                                print("Add module to GuidedBackpropReLU")
                                                print(self.model._modules[idx]._modules[subidx]._modules[sub1idx]._modules[sub2idx]._modules[sub3idx])
                                                self.model._modules[idx]._modules[subidx]._modules[sub1idx]._modules[sub2idx]._modules[sub3idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        if self.cuda:
            output = self.forward(input.cuda())

        else:
            output = self.forward(input)
        softmax = F.softmax(output, dim=1)

        if index is None:
            index = np.argmax(softmax.cpu().data.numpy(), axis=1)

        print("index:", index)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="r2.5d_d34_l32.pth", help="Path to pretrained model")
    parser.add_argument("--video_path", type=str, default="./_EN7WZryBZQ_000690_000700.mp4", help="Path to test video")
    parser.add_argument("--num_classes", type=int, default=400, help="Num classes")
    parser.add_argument("--use_cuda", type=bool, default=False, help="Use GPU acceleration")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    model = r2plus1d_34(num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.checkpoint_path))

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)

    input = get_input_frames(args)

    gb = gb_model(input)

    gb = gb.transpose((1, 2, 3, 0))
    gb = np.mean(gb, axis=0)
    gb = gb - np.min(gb)
    gb = gb / np.max(gb)
    gb = np.uint8(gb * 255)
    gb = cv2.resize(gb, (448, 448))

    cv2.imshow("", gb)
    cv2.waitKey(0)

    cv2.imwrite("guidedbackprop.jpg", gb)

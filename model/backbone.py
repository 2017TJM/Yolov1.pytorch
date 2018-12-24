import torch as t
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, LeakyReLU

class Backbone(Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.stage1 = Sequential()
        self.stage2 = Sequential()
        self.stage3 = Sequential()
        self.stage4 = Sequential()
        self.stage5 = Sequential()

        self.stage1.add_module('stage1_conv1', Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3))
        self.stage1.add_module('stage1_lrelu', LeakyReLU(0.1))
        self.stage1.add_module('stage1_pool', MaxPool2d(kernel_size=(2, 2), stride=2))

        self.stage2.add_module('stage2_conv1', Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), padding=1))
        self.stage2.add_module('stage2_lrelu', LeakyReLU(0.1))
        self.stage2.add_module('stage2_pool', MaxPool2d(kernel_size=(2, 2), stride=2))

        self.stage3.add_module('stage3_conv1', Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1)))
        self.stage3.add_module('stage3_lrelu', LeakyReLU(0.1))
        self.stage3.add_module('stage3_conv2', Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1))
        self.stage3.add_module('stage3_lrelu', LeakyReLU(0.1))
        self.stage3.add_module('stage3_conv3', Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)))
        self.stage3.add_module('stage3_lrelu', LeakyReLU(0.1))
        self.stage3.add_module('stage3_conv4', Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), padding=1))
        self.stage3.add_module('stage3_lrelu', LeakyReLU(0.1))
        self.stage3.add_module('stage3_pool', MaxPool2d(kernel_size=(2, 2), stride=2))

        for i in range(4):
            self.stage4.add_module('stage4_conv%d' % (i*2 + 1), Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)))
            self.stage3.add_module('stage4_lrelu%d' % (i*2 + 1), LeakyReLU(0.1))
            self.stage4.add_module('stage4_conv%d' % (i*2 + 2), Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1))
            self.stage3.add_module('stage4_lrelu%d' % (i*2 + 2), LeakyReLU(0.1))

        self.stage4.add_module('stage4_conv9', Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)))
        self.stage4.add_module('stage4_conv10', Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=1))
        self.stage4.add_module('stage4_pool', MaxPool2d(kernel_size=(2, 2), stride=2))

        for i in range(2):
            self.stage5.add_module('stage5_conv%d' % (i*2 + 1), Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1)))
            self.stage3.add_module('stage5_lrelu%d' % (i*2 + 1), LeakyReLU(0.1))
            self.stage5.add_module('stage5_conv%d' % (i*2 + 2), Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=1))
            self.stage3.add_module('stage5_lrelu%d' % (i*2 + 2), LeakyReLU(0.1))

    def forward(self, x, verbose=False):
        if verbose:
            print('Input shape:', x.shape)
        x = self.stage1(x)
        if verbose:
            print('Stage1 shape:', x.shape)
        x = self.stage2(x)
        if verbose:
            print('Stage2 shape:', x.shape)
        x = self.stage3(x)
        if verbose:
            print('Stage3 shape:', x.shape)
        x = self.stage4(x)
        if verbose:
            print('Stage4 shape:', x.shape)
        x = self.stage5(x)
        if verbose:
            print('Stage5 shape:', x.shape)

        return x
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, conv_kernel,
                 pool_kernel, pool_stride,
                 repeat_blocks, n_filters,
                 batch_norm, dropout,
                 in_channels, num_classes):
        super(UNet, self).__init__()

        self.conv_kernel = conv_kernel
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.repeat_blocks = repeat_blocks
        self.n_filters = n_filters
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.in_channels = in_channels
        self.num_classes = num_classes

    @property
    def down_convs(self):
        down_convs = []
        for i in range(self.repeat_blocks):
            if i == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.in_filters * 2 ** (i - 1)
            down_convs.append(DownConv(in_channels, self.conv_kernel, self.batch_norm, self.conv_dropout))
        return nn.ModuleList(down_convs)

    @property
    def up_convs(self):
        up_convs = []
        for i in range(self.repeat_blocks):
            in_channels = self.in_filters * 2 ** (i + 1)
            up_convs.append(UpConv(in_channels, self.conv_kernel, self.batch_norm, self.conv_dropout))
        return nn.ModuleList(up_convs)

    @property
    def down_pools(self):
        down_pools = []
        for _ in range(self.repeat_blocks):
            down_pools.append(nn.MaxPool2d(kernel_size=(self.pool_kernel, self.pool_kernel),
                                           stride=self.pool_stride))
        return nn.ModuleList(down_pools)

    @property
    def up_samples(self):
        up_samples = []
        for i in range(self.repeat_blocks):
            in_channels = self.in_filters * 2 ** i

            up_samples.append(nn.ConvTranspose2d(in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=(self.pool_kernel, self.pool_kernel),
                                                 stride=self.pool_stride,
                                                 padding=0,
                                                 output_padding=0,
                                                 bias=True))
        return nn.ModuleList(up_samples)

    @property
    def input_layer(self):
        return nn.Conv2d(in_channels=self.in_channels, out_channels=self.n_filters,
                         kernel_size=(self.conv_kernel, self.conv_kernel),
                         stride=1, padding=0, bias=True)

    @property
    def floor_layer(self):
        in_channels = self.n_filters * 2 ** self.repeat_blocks
        return nn.Sequential(DownConv(in_channels, self.conv_kernel, self.batch_norm, self.conv_dropout))

    @property
    def classification_layer(self):
        return nn.Sequential(nn.Conv2d(in_channels=self.n_filters, out_channels=self.num_classes,
                                       kernel_size=(1, 1), stride=1, padding=0, bias=True),
                             nn.Sigmoid()
                             )

    def forward(self, x):

        x = self.input_layer(x)

        down_convs_outputs = []
        for block, down_pool in zip(self.down_convs, self.down_pools):
            x = block(x)
            down_convs_outputs.append(x)
            x = down_pool(x)

        x = self.floor_layer(x)

        for down_conv_output, block, up_sample in zip(reversed(down_convs_outputs),
                                                      reversed(self.up_convs),
                                                      reversed(self.up_samples)):
            x = torch.cat((down_conv_output, x), dim=1)
            x = block(x)
            x = up_sample(x)

        x = self.classification_layer(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, kernel_size, batch_norm, dropout):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout

    @property
    def down_conv(self):
        if self.batch_norm:
            down_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                                kernel_size=(self.conv_kernel, self.conv_kernel),
                                                stride=1, padding=0, bias=True),
                                      nn.BatchNorm2d(num_features=self.in_channels),
                                      nn.Relu(),

                                      nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                                kernel_size=(self.conv_kernel, self.conv_kernel),
                                                stride=1, padding=0, bias=True),
                                      nn.BatchNorm2d(num_features=self.in_channels),
                                      nn.Relu(),

                                      nn.Dropout(self.dropout)
                                      )
        else:
            down_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                                kernel_size=(self.conv_kernel, self.conv_kernel),
                                                stride=1, padding=0, bias=True),
                                      nn.Relu(),

                                      nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                                kernel_size=(self.conv_kernel, self.conv_kernel),
                                                stride=1, padding=0, bias=True),
                                      nn.Relu(),

                                      nn.Dropout(self.dropout)
                                      )
        return down_conv

    def forward(self, x):
        return self.down_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, kernel_size, batch_norm, dropout):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.block_channels = in_channels / 2.
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout

    @property
    def up_conv(self):
        if self.batch_norm:
            up_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.block_channels,
                                              kernel_size=(self.conv_kernel, self.conv_kernel),
                                              stride=1, padding=0, bias=True),
                                    nn.BatchNorm2d(num_features=self.block_channels),
                                    nn.Relu(),

                                    nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_channels,
                                              kernel_size=(self.conv_kernel, self.conv_kernel),
                                              stride=1, padding=0, bias=True),
                                    nn.BatchNorm2d(num_features=self.block_channels),
                                    nn.Relu(),

                                    nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_channels,
                                              kernel_size=(self.conv_kernel, self.conv_kernel),
                                              stride=1, padding=0, bias=True),
                                    nn.BatchNorm2d(num_features=self.block_channels),
                                    nn.Relu(),

                                    nn.Dropout(self.dropout)
                                    )
        else:
            up_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.block_channels,
                                              kernel_size=(self.conv_kernel, self.conv_kernel),
                                              stride=1, padding=0, bias=True),
                                    nn.Relu(),

                                    nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_channels,
                                              kernel_size=(self.conv_kernel, self.conv_kernel),
                                              stride=1, padding=0, bias=True),
                                    nn.Relu(),

                                    nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_channels,
                                              kernel_size=(self.conv_kernel, self.conv_kernel),
                                              stride=1, padding=0, bias=True),
                                    nn.Relu(),

                                    nn.Dropout(self.dropout)
                                    )
        return up_conv

    def forward(self, x):
        return self.up_conv(x)

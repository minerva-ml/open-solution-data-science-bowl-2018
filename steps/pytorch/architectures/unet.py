import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, conv_kernel,
                 pool_kernel, pool_stride,
                 repeat_blocks, n_filters,
                 batch_norm, dropout,
                 in_channels, **kwargs):
        super(UNet, self).__init__()

        self.conv_kernel = conv_kernel
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.repeat_blocks = repeat_blocks
        self.n_filters = n_filters
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.in_channels = in_channels

        self.input_block = self._input_block()
        self.down_convs = self._down_convs()
        self.down_pools = self._down_pools()
        self.floor_block = self._floor_block()
        self.up_convs = self._up_convs()
        self.up_samples = self._up_samples()
        self.classification_block = self._classification_block()
        self.output_layer = self._output_layer()

    def _down_convs(self):
        down_convs = []
        for i in range(self.repeat_blocks):
            in_channels = int(self.n_filters * 2 ** i)
            down_convs.append(DownConv(in_channels, self.conv_kernel, self.batch_norm, self.dropout))
        return nn.ModuleList(down_convs)

    def _up_convs(self):
        up_convs = []
        for i in range(self.repeat_blocks):
            in_channels = int(self.n_filters * 2 ** (i + 2))
            up_convs.append(UpConv(in_channels, self.conv_kernel, self.batch_norm, self.dropout))
        return nn.ModuleList(up_convs)

    def _down_pools(self):
        down_pools = []
        for _ in range(self.repeat_blocks):
            down_pools.append(nn.Sequential(nn.ConstantPad2d(self.pool_kernel-2, 0), #this padding works for pool_kernel=2,3
                                            nn.MaxPool2d(kernel_size=(self.pool_kernel, self.pool_kernel),
                                                         stride=self.pool_stride)))
        return nn.ModuleList(down_pools)

    def _up_samples(self):
        up_samples = []
        for i in range(self.repeat_blocks):
            in_channels = int(self.n_filters * 2 ** (i + 2))
            out_channels = int(self.n_filters * 2 ** (i + 1))
            up_samples.append(nn.ConvTranspose2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=2,
                                                 stride=2,
                                                 padding=0,
                                                 output_padding=0,
                                                 bias=False
                                                 ))
        return nn.ModuleList(up_samples)

    def _input_block(self):
        if self.batch_norm:
            input_block = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=1),
                                        nn.BatchNorm2d(num_features=self.n_filters),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=1),
                                        nn.BatchNorm2d(num_features=self.n_filters),
                                        nn.ReLU(),

                                        nn.Dropout(self.dropout),
                                        )
        else:
            input_block = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=1),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=1),
                                        nn.ReLU(),

                                        nn.Dropout(self.dropout),
                                        )
        return input_block

    def _floor_block(self):
        in_channels = int(self.n_filters * 2 ** self.repeat_blocks)
        return nn.Sequential(DownConv(in_channels, self.conv_kernel, self.batch_norm, self.dropout),
                             )

    def _classification_block(self):
        in_block = int(2 * self.n_filters)

        if self.batch_norm:
            classification_block = nn.Sequential(nn.Conv2d(in_channels=in_block, out_channels=self.n_filters,
                                                           kernel_size=(self.conv_kernel, self.conv_kernel),
                                                           stride=1, padding=1),
                                                 nn.BatchNorm2d(num_features=self.n_filters),
                                                 nn.ReLU(),
                                                 nn.Dropout(self.dropout),

                                                 nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters,
                                                           kernel_size=(self.conv_kernel, self.conv_kernel),
                                                           stride=1, padding=1),
                                                 nn.BatchNorm2d(num_features=self.n_filters),
                                                 nn.ReLU(),
                                                 )
        else:
            classification_block = nn.Sequential(nn.Conv2d(in_channels=in_block, out_channels=self.n_filters,
                                                           kernel_size=(self.conv_kernel, self.conv_kernel),
                                                           stride=1, padding=1),
                                                 nn.ReLU(),
                                                 nn.Dropout(self.dropout),

                                                 nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters,
                                                           kernel_size=(self.conv_kernel, self.conv_kernel),
                                                           stride=1, padding=1),
                                                 nn.ReLU(),
                                                 )
        return classification_block

    def _output_layer(self):
        return nn.Conv2d(in_channels=self.n_filters, out_channels=1,
                         kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        x = self.input_block(x)
        import pdb;pdb.set_trace()

        down_convs_outputs = []
        for block, down_pool in zip(self.down_convs, self.down_pools):
            x = block(x)
            down_convs_outputs.append(x)
            x = down_pool(x)
        x = self.floor_block(x)

        for down_conv_output, block, up_sample in zip(reversed(down_convs_outputs),
                                                      reversed(self.up_convs),
                                                      reversed(self.up_samples)):
            x = up_sample(x)
            x = torch.cat((down_conv_output, x), dim=1)

            x = block(x)

        x = self.classification_block(x)
        x = self.output_layer(x)
        return x


class UNetMultitask(UNet):
    def __init__(self, conv_kernel,
                 pool_kernel, pool_stride,
                 repeat_blocks, n_filters,
                 batch_norm, dropout,
                 in_channels, nr_outputs):
        super(UNetMultitask, self).__init__(conv_kernel,
                                            pool_kernel, pool_stride,
                                            repeat_blocks, n_filters,
                                            batch_norm, dropout,
                                            in_channels)

        self.nr_outputs = nr_outputs
        output_legs = []
        for i in range(self.nr_outputs):
            output_legs.append(self._output_layer())
        self.output_legs = nn.ModuleList(output_legs)

    def forward(self, x):
        x = self.input_block(x)

        down_convs_outputs = []
        for block, down_pool in zip(self.down_convs, self.down_pools):
            x = block(x)
            down_convs_outputs.append(x)
            x = down_pool(x)
        x = self.floor_block(x)

        for down_conv_output, block, up_sample in zip(reversed(down_convs_outputs),
                                                      reversed(self.up_convs),
                                                      reversed(self.up_samples)):
            x = up_sample(x)
            x = torch.cat((down_conv_output, x), dim=1)

            x = block(x)

        x = self.classification_block(x)

        outputs = [output_leg(x) for output_leg in self.output_legs]
        return outputs

class DCAN(UNet):
    def __init__(self, conv_kernel,
                 pool_kernel, pool_stride,
                 repeat_blocks, n_filters,
                 batch_norm, dropout,
                 in_channels, n_classifiers):
        assert conv_kernel%2==1
        self.n_classifiers = n_classifiers
        super(DCAN, self).__init__(conv_kernel,
                                            pool_kernel, pool_stride,
                                            repeat_blocks, n_filters,
                                            batch_norm, dropout,
                                            in_channels)

        self.convs_for_classifiers = self._convs_for_classifiers()
        self.down_pools_for_classifiers = self._down_pools_for_classifiers()
        self.mask_output_layer = self._mask_output_layer()
        self.contour_output_layer = self._contour_output_layer()
        self.up_convs_mask = self._up_samples()
        self.up_convs_contour = self._up_samples()
        self.last_block = self._last_block()
        self.last_up_conv_mask = self._last_up_conv()
        self.last_up_conv_contour = self._last_up_conv()
        self.mask_to_single_channel = self._to_single_channel()
        self.contour_to_single_channel = self._to_single_channel()

    def _to_single_channel(self):
        in_channels = int(self.n_filters * 2 ** self.repeat_blocks)
        out_channels = 1
        return nn.ModuleList([Conv(in_channels, out_channels, self.conv_kernel, self.batch_norm, self.dropout) for i in range(self.n_classifiers)])

    def _last_up_conv(self):
        in_channels = int(self.n_filters * 2 ** (self.repeat_blocks + 1))
        out_channels = int(in_channels/2)
        up_conv = nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=self.pool_stride ** (self.repeat_blocks + self.n_classifiers-1),
                                             stride=self.pool_stride ** (self.repeat_blocks + self.n_classifiers-1),
                                             padding=0,
                                             output_padding=0,
                                             bias=False
                                             )
        return up_conv

    def _last_block(self):
        in_channels = self.n_filters * 2 ** self.repeat_blocks
        return DownConv(in_channels, self.conv_kernel, self.batch_norm, self.dropout)

    def _input_block(self):
        if self.batch_norm:
            input_block = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=int(self.conv_kernel/2)),
                                        nn.BatchNorm2d(num_features=self.n_filters),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=int(self.conv_kernel/2)),
                                        nn.BatchNorm2d(num_features=self.n_filters),
                                        nn.ReLU(),

                                        nn.Dropout(self.dropout),
                                        )
        else:
            input_block = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=int(self.conv_kernel/2)),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=int(self.conv_kernel/2)),
                                        nn.ReLU(),

                                        nn.Dropout(self.dropout),
                                        )
        return input_block

    def _output_block(self):
        in_channels = int(self.n_filters * 2 ** self.repeat_blocks)
        out_channels = 1
        if self.batch_norm:
            output_block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=int(self.conv_kernel/2)),
                                        nn.BatchNorm2d(num_features=self.n_filters),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=self.n_filters, out_channels=out_channels,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=int(self.conv_kernel/2)),
                                        nn.BatchNorm2d(num_features=self.n_filters),
                                        nn.ReLU(),

                                        nn.Dropout(self.dropout),
                                        )
        else:
            output_block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=self.n_filters,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=int(self.conv_kernel/2)),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=self.n_filters, out_channels=self.out_channels,
                                                  kernel_size=(self.conv_kernel, self.conv_kernel),
                                                  stride=1, padding=int(self.conv_kernel/2)),
                                        nn.ReLU(),

                                        nn.Dropout(self.dropout),
                                        )
        return output_block

    def _mask_output_layer(self):
        in_channels = int(self.n_filters * 2 ** self.repeat_blocks)
        out_channels = in_channels
        return Conv(in_channels, out_channels, self.conv_kernel, True, self.dropout)

    def _contour_output_layer(self):
        return self._mask_output_layer()

    def _down_pools_for_classifiers(self):
        down_pools = []
        for _ in range(self.repeat_blocks):
            down_pools.append(nn.Sequential(nn.ConstantPad2d(self.pool_kernel-2, 0), #this padding works for pool_kernel=2,3
                                            nn.MaxPool2d(kernel_size=(self.pool_kernel, self.pool_kernel),
                                                         stride=self.pool_stride)))
        return nn.ModuleList(down_pools)

    def _convs_for_classifiers(self):
        down_convs = []
        for i in range(self.n_classifiers-1):
            in_channels = self.n_filters * 2 ** self.repeat_blocks
            down_convs.append(Conv(in_channels, in_channels, self.conv_kernel, self.batch_norm, self.dropout))
        return nn.ModuleList(down_convs)

    def _up_samples(self):
        up_samples = []
        for i in range(self.n_classifiers-1):
            in_channels = int(self.n_filters * 2 ** self.repeat_blocks)
            out_channels = in_channels
            up_samples.append(nn.ConvTranspose2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=self.pool_stride ** (self.repeat_blocks + i),
                                                 stride=self.pool_stride ** (self.repeat_blocks + i),
                                                 padding=0,
                                                 output_padding=0,
                                                 bias=False
                                                 ))
        return nn.ModuleList(up_samples)

    def forward(self, x):
        x = self.input_block(x)

        down_convs_outputs = []
        for block, down_pool in zip(self.down_convs, self.down_pools):
            x = block(x)
            down_convs_outputs.append(x)
            x = down_pool(x)

        down_convs_outputs_for_classifiers = []
        for block, down_pool in zip(self.convs_for_classifiers, self.down_pools_for_classifiers):
            x = block(x)
            down_convs_outputs_for_classifiers.append(x)
            x = down_pool(x)

        x = self.last_block(x)
        mask = self.last_up_conv_mask(x)
        mask = self.mask_output_layer(mask)

        contour = self.last_up_conv_contour(x)
        contour = self.contour_output_layer(contour)

        mask_classifier_inputs = []
        for down_convs_output, up_conv in zip(down_convs_outputs_for_classifiers, self.up_convs_mask):
            tmp = up_conv(down_convs_output)
            mask_classifier_inputs.append(tmp)
        mask_classifier_inputs.append(mask)

        contour_classifier_inputs = []
        for down_convs_output, up_conv in zip(down_convs_outputs_for_classifiers, self.up_convs_contour):
            tmp = up_conv(down_convs_output)
            contour_classifier_inputs.append(tmp)
        contour_classifier_inputs.append(contour)

        tmp = []
        for input, to_single in zip(mask_classifier_inputs, self.mask_to_single_channel):
            tmp.append(to_single(input))
        mask_classifier_inputs = tmp

        tmp = []
        for input, to_single in zip(contour_classifier_inputs, self.contour_to_single_channel):
            tmp.append(to_single(input))
        contour_classifier_inputs = tmp

        mask_sum = torch.cat(mask_classifier_inputs, dim=1)
        mask_weights = torch.autograd.Variable(torch.randn(self.n_classifiers), requires_grad=True)
        if torch.cuda.is_available():
            mask_weights = mask_weights.cuda()
        mask_sum = torch.cat([mask_sum[:, c_i, :, :].contiguous().view_as(mask_classifier_inputs[0])*d_i
                              for c_i,d_i in zip(range(self.n_classifiers), mask_weights)], dim=1)
        mask_sum = torch.sum(mask_sum, 1).view_as(mask_classifier_inputs[0])

        contour_sum = torch.cat(contour_classifier_inputs, dim=1)
        contour_weights = torch.autograd.Variable(torch.randn(self.n_classifiers), requires_grad=True)
        if torch.cuda.is_available():
            contour_weights = contour_weights.cuda()
        contour_sum = torch.cat([contour_sum[:,c_i, :, :].contiguous().view_as(contour_classifier_inputs[0])*d_i
                                 for c_i,d_i in zip(range(self.n_classifiers), contour_weights)], dim=1)
        contour_sum = torch.sum(contour_sum, 1).view_as(contour_classifier_inputs[0])

        output = [mask_sum, contour_sum, mask_classifier_inputs, contour_classifier_inputs]

        return output


class DownConv(nn.Module):
    def __init__(self, in_channels, kernel_size, batch_norm, dropout):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.block_channels = int(in_channels * 2.)
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.down_conv = self._down_conv()

    def _down_conv(self):
        if self.batch_norm:
            down_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.block_channels,
                                                kernel_size=(self.kernel_size, self.kernel_size),
                                                stride=1, padding=int(self.kernel_size/2)),
                                      nn.BatchNorm2d(num_features=self.block_channels),
                                      nn.ReLU(),

                                      nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_channels,
                                                kernel_size=(self.kernel_size, self.kernel_size),
                                                stride=1, padding=int(self.kernel_size/2)),
                                      nn.BatchNorm2d(num_features=self.block_channels),
                                      nn.ReLU(),

                                      nn.Dropout(self.dropout),
                                      )
        else:
            down_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.block_channels,
                                                kernel_size=(self.kernel_size, self.kernel_size),
                                                stride=1, padding=int(self.kernel_size/2)),
                                      nn.ReLU(),

                                      nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_channels,
                                                kernel_size=(self.kernel_size, self.kernel_size),
                                                stride=1, padding=int(self.kernel_size/2)),
                                      nn.ReLU(),

                                      nn.Dropout(self.dropout),
                                      )
        return down_conv

    def forward(self, x):
        return self.down_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, kernel_size, batch_norm, dropout):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.block_channels = int(in_channels / 2.)
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.up_conv = self._up_conv()

    def _up_conv(self):
        if self.batch_norm:
            up_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.block_channels,
                                              kernel_size=(self.kernel_size, self.kernel_size),
                                              stride=1, padding=int(self.kernel_size/2)),

                                    nn.BatchNorm2d(num_features=self.block_channels),
                                    nn.ReLU(),

                                    nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_channels,
                                              kernel_size=(self.kernel_size, self.kernel_size),
                                              stride=1, padding=int(self.kernel_size/2)),
                                    nn.BatchNorm2d(num_features=self.block_channels),
                                    nn.ReLU(),

                                    nn.Dropout(self.dropout)
                                    )
        else:
            up_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.block_channels,
                                              kernel_size=(self.kernel_size, self.kernel_size),
                                              stride=1, padding=int(self.kernel_size/2)),
                                    nn.ReLU(),

                                    nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_channels,
                                              kernel_size=(self.kernel_size, self.kernel_size),
                                              stride=1, padding=int(self.kernel_size/2)),
                                    nn.ReLU(),

                                    nn.Dropout(self.dropout)
                                    )
        return up_conv

    def forward(self, x):
        return self.up_conv(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm, dropout):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.conv = self._conv()

    def _conv(self):
        if self.batch_norm:
            conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                              kernel_size=(self.kernel_size, self.kernel_size),
                                              stride=1, padding=int(self.kernel_size/2)),

                                    nn.BatchNorm2d(num_features=self.out_channels),
                                    nn.ReLU(),

                                    nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                                              kernel_size=(self.kernel_size, self.kernel_size),
                                              stride=1, padding=int(self.kernel_size/2)),
                                    nn.BatchNorm2d(num_features=self.out_channels),
                                    nn.ReLU(),

                                    nn.Dropout(self.dropout)
                                    )
        else:
            conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                              kernel_size=(self.kernel_size, self.kernel_size),
                                              stride=1, padding=int(self.kernel_size/2)),
                                    nn.ReLU(),

                                    nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                                              kernel_size=(self.kernel_size, self.kernel_size),
                                              stride=1, padding=int(self.kernel_size/2)),
                                    nn.ReLU(),

                                    nn.Dropout(self.dropout)
                                    )
        return conv

    def forward(self, x):
        return self.conv(x)

if __name__=="__main__":
    x = torch.randn([2,4,512,512])
    x = torch.autograd.Variable(x)
    dcan = DCAN(7, 3, 2, 2, 16, True, 0, 4, 1, 3, 3)
    x0 = x
    x = dcan(x)
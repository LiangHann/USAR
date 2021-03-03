import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(noise_nc, input_nc, output_nc, ngf, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = ResnetGenerator(noise_nc, input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
   
    return init_net(net, init_type, init_gain, gpu_id)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, noise_nc, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        # parameters and operations for U-Net
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)

        self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.sk1 = Conv(ngf * 2, ngf * 2, norm_layer, use_bias)
        self.up2 = Up(ngf * 4, ngf, norm_layer, use_bias)
        self.sk2 = Conv(ngf, ngf, norm_layer, use_bias)

        self.outc = Outconv(ngf * 2, output_nc)

        # parameters and operations for background generator
        self.linear = nn.Linear(noise_nc, 2048, bias=True)
        self.layer_up_1 = ResidualBlock(128, 128, 'up', up_size=8)
        self.layer_up_2 = ResidualBlock(128, 128, 'up', up_size=16)
        self.layer_up_3 = ResidualBlock(128, 128, 'up', up_size=32)
        self.layer_up_4 = ResidualBlock(128, 128, 'up', up_size=64)
        self.layer_up_5 = ResidualBlock(128, 128, 'up', up_size=128)
        self.layer_up_6 = ResidualBlock(128, 128, 'up', up_size=256)
        self.layer_up_7 = ResidualBlock(128, 128, 'up', up_size=512)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv_last = nn.Conv2d(128, output_nc, 3, 1, 1, bias=True)

    def forward(self, noise_vec, input_img):
        ######################
        # (1) U-Net for generating mask of the input image
        out = {}
        out['in'] = self.inc(input_img) # 64*256*256
        out['d1'] = self.down1(out['in']) # 128*128*128
        out['d2'] = self.down2(out['d1']) # 256*64*64
        out['bottle'] = self.resblocks(out['d2']) # 256*64*64


        out['u1'] = self.up1(out['bottle'])
        out['s1'] = self.sk1(out['d1'])
        out['c1'] = torch.cat((out['s1'], out['u1']), 1)

        out['u2'] = self.up2(out['c1'])
        out['s2'] = self.sk2(out['in'])
        out['c2'] = torch.cat((out['s2'], out['u2']), 1)

        ######################
        # (2) Res-Blocks for generating background image
        x = self.linear(noise_vec)
        x = x.view(-1, 128, 4, 4)
        x = self.layer_up_1(x)
        x = self.layer_up_2(x)
        x = self.layer_up_3(x)
        x = self.layer_up_4(x)
        x = self.layer_up_5(x)
        x = self.layer_up_6(x)
        x = self.layer_up_7(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv_last(x)
        x = nn.functional.tanh(x)

        return self.outc(out['c2']), x


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
#            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size = 7, padding = 3,
                      bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3, stride=1,
                      padding=1, bias=use_bias),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.ConvTranspose2d(in_ch, out_ch,
            #                    kernel_size=3, stride=2,
            #                    padding=1, output_padding=1,
            #                    bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Conv, self).__init__()
        self.con = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=1, stride=1,
                      padding=0, bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.con(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
#            nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, resample=None, up_size=0, down_size=0):
        super(ResidualBlock, self).__init__()
        if resample == 'up':
            self.bn1 = nn.BatchNorm2d(in_dim)
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.upsample = torch.nn.Upsample(scale_factor = 2)
            self.upsample_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.bn2 = nn.BatchNorm2d(out_dim)
        elif resample == 'down':
            self.bn1 = nn.BatchNorm2d(in_dim)
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.downsample = nn.AvgPool2d(2)
            self.downsample_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.resample = resample

    def forward(self, x):
        if self.resample == 'up':
            shortcut = x
            output = x
            
            shortcut = self.upsample(shortcut) #upsampleconv
            shortcut = self.upsample_conv(shortcut)
            
            output = self.bn1(output)
            output = nn.functional.relu(output)
            output = self.conv1(output)

            output = self.bn2(output)
            output = nn.functional.relu(output)
            output = self.upsample(output) #upsampleconv
            output = self.conv2(output)
            
        elif self.resample == 'down':
            shortcut = x
            output = x
            
            shortcut = self.downsample(shortcut)      #dowmsample
            shortcut = self.downsample_conv(shortcut) #dowmsampleconv
            
            output = self.conv1(output)
            output = self.bn1(output)
            output = nn.functional.relu(output)

            output = self.downsample(output)          #downsampleconv
            output = self.conv2(output)
            output = self.bn2(output)
            output = nn.functional.relu(output)

        return output+shortcut


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='instance', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'patch':
        net = PatchDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_id)


def define_Content(input_nc, ndf,
                   n_layers_D=3, norm='instance', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    return init_net(net, init_type, init_gain, gpu_id)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]
        
        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    norm_layer(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    norm_layer(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    norm_layer(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        if use_sigmoid:
            model += [nn.Sigmoid()]
        

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x =  self.model(input)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PatchDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class WGANGP(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(WGANGP, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def define_Z(noise_nc, input_nc, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):

    net = netRecZ(noise_nc, input_nc)
   
    return init_net(net, init_type, init_gain, gpu_id)


class netRecZ(nn.Module):
    def __init__(self, noise_nc, input_nc):
        super(netRecZ, self).__init__()
        self.conv_first = nn.Conv2d(input_nc, 128, 3, 1, 1, bias=True)
        self.layer_down_1 = ResidualBlock(128, 128, 'down', down_size=128)
        self.layer_down_2 = ResidualBlock(128, 128, 'down', down_size=64)
        self.layer_down_3 = ResidualBlock(128, 128, 'down', down_size=32)
        self.layer_down_4 = ResidualBlock(128, 128, 'down', down_size=16)
        self.layer_down_5 = ResidualBlock(128, 128, 'down', down_size=8)
        self.layer_down_6 = ResidualBlock(128, 128, 'down', down_size=4)
        self.bn1 = nn.BatchNorm2d(128)
        self.linear = nn.Linear(2048, noise_nc, bias=True)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.layer_down_1(x)
        x = self.layer_down_2(x)
        x = self.layer_down_3(x)
        x = self.layer_down_4(x)
        x = self.layer_down_5(x)
        x = self.layer_down_6(x)
        x = x.view(-1, 2048)
        x = self.linear(x)

        return x

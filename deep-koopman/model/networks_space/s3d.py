import torch.nn as nn
import torch

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class STConv3d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0):
        super(STConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size),stride=(1,stride,stride),padding=(0,padding,padding))
        self.conv2 = nn.Conv3d(out_planes,out_planes,kernel_size=(kernel_size,1,1),stride=(1,1,1),padding=(padding,0,0))

        self.bn=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.bn2=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu2=nn.ReLU(inplace=True)

        nn.init.normal(self.conv2.weight,mean=0,std=0.01)
        nn.init.constant(self.conv2.bias,0)

    def forward(self,x):
        x=self.conv(x)
        #x=self.conv2(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        return x

# Note the operations here for S3D-G:
# If we set two convs: 1xkxk + kx1x1, it's as follows: (p=(k-1)/2)
# BasicConv3d(input,output,kernel_size=(1,k,k),stride=1,padding=(0,p,p))
# Then BasicConv3d(output,output,kernel_size=(k,1,1),stride=1,padding=(p,0,0))

class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            STConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            STConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            STConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            STConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            STConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            STConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            STConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            STConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            STConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            STConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            STConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            STConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_4_last(nn.Module):
    def __init__(self):
        super(Mixed_4_last, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
            STConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            STConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )

class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            STConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            STConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            STConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            STConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            STConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            STConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class S3DG(nn.Module):

    def __init__(self, num_classes=400, dropout_keep_prob = 1, input_channel = 3, spatial_squeeze=True):
        super(S3DG, self).__init__()
        self.features = nn.Sequential(
            STConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3), # (64, 32, 112, 112)
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (64, 32, 56, 56)
            BasicConv3d(64, 64, kernel_size=1, stride=1), # (64, 32, 56, 56)
            STConv3d(64, 192, kernel_size=3, stride=1, padding=1),  # (192, 32, 56, 56)
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (192, 32, 28, 28)
            Mixed_3b(), # (256, 32, 28, 28)
            Mixed_3c(), # (480, 32, 28, 28)
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), # (480, 16, 14, 14)
            Mixed_4b(),# (512, 16, 14, 14)
            Mixed_4c(),# (512, 16, 14, 14)
            Mixed_4d(),# (512, 16, 14, 14)
            Mixed_4e(),# (528, 16, 14, 14)
            Mixed_4f(),# (832, 16, 14, 14)
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # (832, 8, 7, 7)
            Mixed_5b(), # (832, 8, 7, 7)
            Mixed_5c(), # (1024, 8, 7, 7)
            nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1),# (1024, 8, 1, 1)
            nn.Dropout3d(dropout_keep_prob),
            nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),# (400, 8, 1, 1)
        )
        self.spatial_squeeze = spatial_squeeze
        self.softmax = nn.Softmax()

    def forward(self, x):
        logits = self.features(x)

        if self.spatial_squeeze:
            logits = logits.squeeze(3)
            logits = logits.squeeze(3)

        averaged_logits = torch.mean(logits, 2)
        predictions = self.softmax(averaged_logits)

        return predictions, averaged_logits

''' Modified and simplified S3D'''

class STMessagePassing(nn.Module):
    def __init__(self, in_chan, hidden_dim, out_chan, kernel, stride, padding, n_messages = 3):
        super(STMessagePassing, self).__init__()
        self.objs = kernel[-1] * kernel[-2]
        self.n_messages = n_messages

        layers = [BasicConv3d(in_chan, hidden_dim * self.objs , kernel_size=kernel, stride=stride, padding=padding)]
        layers.append(Reshape(self.objs, hidden_dim))
        for i in range(n_messages - 2):
            layers.append(BasicConv3d(hidden_dim, hidden_dim * self.objs, kernel_size=kernel, stride=stride, padding=padding))
            layers.append(Reshape(self.objs, hidden_dim))
        layers.append(nn.Conv3d(hidden_dim, out_chan * self.objs, kernel_size=kernel, stride=stride, padding=padding))
        layers.append(Reshape(self.objs, out_chan))

        self.messages = nn.Sequential(*layers)

    def forward(self, x):
        x = self.messages(x)
        return x

class Reshape(nn.Module):
    def __init__(self, chans, objs):
        super(Reshape, self).__init__()
        self.objs = objs
        self.chans = chans

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.objs, x.shape[-3]).permute(0,1,3,2).reshape(x.shape[0], self.chans, -1, int(torch.sqrt(self.objs)), int(torch.sqrt(self.objs)))
        print(x.shape)
        return x

class Simple_S3D(nn.Module):

    def __init__(self, chan_out=1, n_obj=1, dropout_keep_prob = 1, input_channel = 1, spatial_squeeze=True):
        super(Simple_S3D, self).__init__()
        self.features = nn.Sequential(
            STConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3), # (64, 32, 32, 32)
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (64, 32, 16, 16) (Makes sense?)
            STConv3d(64, 64, kernel_size=5, stride=1, padding=2), # (64, 32, 32, 32)
            STConv3d(64, 64, kernel_size=3, stride=1, padding=1), # (64, 32, 32, 32)
            BasicConv3d(64, 64, kernel_size=1, stride=1), # (64, 32, 16, 16)
            STConv3d(64, 192, kernel_size=3, stride=1, padding=1),  # (192, 32, 16, 16)
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (192, 32, 8, 8) (Makes sense?)
            Mixed_3b(), # (256, 32, 28, 28)
            Mixed_3c(), # (480, 32, 28, 28)
            # nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), # (480, 16, 14, 14)
            Mixed_4b(),# (512, 16, 14, 14)
            # PrintShape(),
            # Mixed_4c(),# (512, 16, 14, 14)
            # Mixed_4d(),# (512, 16, 14, 14)
            # Mixed_4e(),# (528, 16, 14, 14)
            # Mixed_4f(),# (528, 16, 14, 14)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # (832, 32, 4, 4)
            nn.Dropout3d(dropout_keep_prob),
            # STMessagePassing(512, 512, chan_out, kernel=[5, 4, 4], stride=1, padding=[3,0,0], n_messages = 4),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # (832, 8, 7, 7)
            # Mixed_5b(), # (832, 8, 7, 7)
            # PrintShape(),
            # Mixed_5c(), # (1024, 8, 7, 7)
            BasicConv3d(512, 512, kernel_size=(1, 4, 4), stride=1, padding=0), # (832, 512, 32, 1, 1)
            # BasicConv3d(832, 512, kernel_size=(1, 4, 4), stride=1, padding=0), # (832, 512, 32, 1, 1)
            # nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1),# (1024, 8, 1, 1)
            nn.Conv3d(512, chan_out * n_obj, kernel_size=1, stride=1, bias=True),# (400, chan*obj, 32, 1, 1)
        )
        self.spatial_squeeze = spatial_squeeze
        # self.softmax = nn.Softmax()

    def forward(self, x):
        logits = self.features(x)
        if self.spatial_squeeze:
            logits = logits.squeeze(3)
            logits = logits.squeeze(3)

        return logits


class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()
    def forward(self,x):
        print(x.shape)
        return x
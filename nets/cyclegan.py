import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class generator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, n_residual_blocks=9):
        super(generator, self).__init__()

        # Initial convolution block       
        self.stage_1 = nn.Sequential(
            *[
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channel, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ]
        )

        self.stage_2 = nn.Sequential(
            *[
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True) 
            ]
        )
        
        self.stage_3 = nn.Sequential(
            *[
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True) 
            ]
        )
        
        # Residual blocks
        self.stage_4 = []
        for _ in range(n_residual_blocks):
            self.stage_4 += [ResidualBlock(256)]
        self.stage_4 = nn.Sequential(*self.stage_4)

        self.up_stage_1 = nn.Sequential(
            *[
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True)
            ]
        )
        
        self.up_stage_2 = nn.Sequential(
            *[
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            ]
        )
        
        self.head = nn.Sequential(
            *[
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, out_channel, 7),
                nn.Tanh() 
            ]
        )
        
        self.weight_init()
        
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.up_stage_1(x)
        x = self.up_stage_2(x)
        return self.head(x)

class discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(discriminator, self).__init__()

        self.stage_1 = nn.Sequential(
            *[
                nn.Conv2d(in_channel, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        )

        self.stage_2 = nn.Sequential(
            *[
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128), 
                nn.LeakyReLU(0.2, inplace=True)
            ]
        )
        
        self.stage_3 = nn.Sequential(
            *[
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.InstanceNorm2d(256), 
                nn.LeakyReLU(0.2, inplace=True)
            ]
        )
        
        self.stage_4 = nn.Sequential(
            *[
                nn.Conv2d(256, 512, 4, padding=1),
                nn.InstanceNorm2d(512), 
                nn.LeakyReLU(0.2, inplace=True)
            ]
        )
        
        self.head = nn.Sequential(
            *[
                nn.Conv2d(512, 1, 4, padding=1),
                nn.AdaptiveAvgPool2d([1, 1])
            ]
        )
        
        self.weight_init()
        
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.head(x)

        return x.view(x.size()[0])

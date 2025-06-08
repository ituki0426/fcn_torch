import torch.nn as nn
class FCN32s(nn.Module):
    def __init__(self, pre_net,n_class):
        super(FCN32s, self).__init__()
        self.pre_net = pre_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels = 512, # 入力チャンネル数
            out_channels = 512, # 出力チャンネル数
            kernel_size=4, # カーネルサイズ
            stride=2, # ストライド
            padding=1 # パディング
        )
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels = 512, # 入力チャンネル数
            out_channels = 256, # 出力チャンネル数
            kernel_size=2, # カーネルサイズ
            padding=1, # パディング
            dilation = 1, # 拡張率
            output_padding = 1, # 出力パディング
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            in_channels = 256, # 入力チャンネル数
            out_channels = 128, # 出力チャンネル数
            kernel_size=3, # カーネルサイズ
            stride=2, # ストライド
            padding=1, # パディング
            dilation = 1, # 拡張率
            output_padding = 1 # 出力パディング
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            in_channels = 128, # 入力チャンネル数
            out_channels = 64, # 出力チャンネル数
            kernel_size = 3, # カーネルサイズ
            stride = 2, # ストライド
            padding = 1, # パディング
            dilation = 1, # 拡張率
            output_padding = 1 # 出力パディング
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            in_channels = 64, # 入力チャンネル数
            out_channels = 32, # 出力チャンネル数
            kernel_size = 3, # カーネルサイズ
            stride = 2, # ストライド
            padding = 1 # パディング
            dilation = 1 # 拡張率
            output_padding = 1 # 出力パディング
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(
            in_channels = n_class, # 入力チャンネル数
            out_channels = n_class, # 出力チャンネル数
            kernel_size=1, # カーネルサイズ
            stride=1, # ストライド
            padding=0 # パディング
        )
    def forward(self, x):
        output = self.pre_net(x)
        x5 = output['x5']

        score = self.bn1(self.relu(self.deconv1(x5)))
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score
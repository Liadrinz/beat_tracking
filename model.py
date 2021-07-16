from typing import List, Tuple
import torch
from torch import nn
from torchvision.models import resnet101, resnext101_32x8d


class PointBeatModel(nn.Module):

    def __init__(self, n_frames, n_features, n_channels, n_output):
        super().__init__()
        self.resnet = resnext101_32x8d()
        self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.n_frames = n_frames
        self.n_features = n_features
        self.n_channels = n_channels
        self.n_output = n_output
        self.fc1 = nn.Linear(self.resnet.fc.out_features, 256)
        self.fc2 = nn.Linear(256, n_output)
        self.bn_fc = nn.BatchNorm1d(256)
    
    def feature(self, X: torch.Tensor):
        X = X.reshape(-1, self.n_channels, self.n_frames, self.n_features)
        X = self.resnet(X)
        return X

    def forward(self, X: torch.Tensor):
        X = self.feature(X)
        X = self.fc1(X)
        X = self.bn_fc(X)
        X = self.fc2(X)
        outputs = torch.sigmoid(X)
        return outputs


class FastSlowResNeXtBeatModel(nn.Module):

    def __init__(self, n_frames, n_features, n_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.fast_model = ResNeXtBeatModel(n_frames, n_features, n_channels, output_shape)
        self.slow_model = ResNeXtBeatModel(n_frames * 2, n_features, n_channels, output_shape)
        self.out_size = output_shape[0] * output_shape[1]
        self.fc1 = nn.Linear(self.fast_model.resnet.fc.out_features * 2, 256)
        self.fc2 = nn.Linear(256, self.output_shape[0] * self.output_shape[1])
        self.bn_fc = nn.BatchNorm1d(256)
    
    def forward(self, X: Tuple[torch.Tensor]):
        X_fast, X_slow = X
        X_fast = self.fast_model.feature(X_fast)
        X_slow = self.slow_model.feature(X_slow)
        X = torch.cat((X_fast, X_slow), dim=-1)

        X = self.fc1(X)
        X = self.bn_fc(X)
        X = self.fc2(X)
        X = torch.sigmoid(X)
        
        outputs = X.reshape(-1, *self.output_shape)
        return outputs


class ResNeXtBeatModel(nn.Module):

    def __init__(self, n_frames, n_features, n_channels, output_shape):
        super().__init__()
        self.resnet = resnext101_32x8d()
        self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.n_frames = n_frames
        self.n_features = n_features
        self.n_channels = n_channels
        self.output_shape = output_shape
        self.fc1 = nn.Linear(self.resnet.fc.out_features, 256)
        self.fc2 = nn.Linear(256, self.output_shape[0] * self.output_shape[1])
        self.bn_fc = nn.BatchNorm1d(256)

    def feature(self, X: torch.Tensor):
        X = X.reshape(-1, self.n_channels, self.n_frames, self.n_features)
        X = self.resnet(X)
        return X

    def forward(self, X: torch.Tensor):
        X = self.feature(X)
        X = self.fc1(X)
        X = self.bn_fc(X)
        X = self.fc2(X)
        X = torch.sigmoid(X)
        
        outputs = X.reshape(-1, *self.output_shape)
        return outputs


class ResNetBeatModel(nn.Module):

    def __init__(self, n_frames, n_features, n_channels, output_shape):
        super().__init__()
        self.resnet = resnet101()
        self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.n_frames = n_frames
        self.n_features = n_features
        self.n_channels = n_channels
        self.output_shape = output_shape
        self.fc1 = nn.Linear(self.resnet.fc.out_features, 256)
        self.fc2 = nn.Linear(256, self.output_shape[0] * self.output_shape[1])
        self.bn_fc = nn.BatchNorm1d(256)
    
    def forward(self, X: torch.Tensor):
        X = X.reshape(-1, self.n_channels, self.n_frames, self.n_features)
        X = self.resnet(X)
        X = self.fc1(X)
        X = self.bn_fc(X)
        X = self.fc2(X)
        X = torch.sigmoid(X)
        
        outputs = X.reshape(-1, *self.output_shape)
        return outputs


class CNNBeatModel(nn.Module):

    def __init__(self, n_frames, n_features, n_channels, output_shape):
        super().__init__()

        self.n_frames = n_frames
        self.n_features = n_features
        self.n_channels = n_channels
        self.output_shape = output_shape

        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 2))
        self.conv2 = nn.Conv2d(64, 64, (5, 5), (1, 1))
        self.conv3 = nn.Conv2d(64, 96, (5, 5), (1, 1))
        self.conv4 = nn.Conv2d(96, 64, (3, 3), (1, 1))
        self.conv5 = nn.Conv2d(64, 24, (3, 3), (1, 1))

        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(24)
        self.bn6 = nn.BatchNorm1d(256)

        self.drop1 = nn.Dropout2d()
        self.drop2 = nn.Dropout2d()
        self.drop3 = nn.Dropout2d()
        self.drop4 = nn.Dropout2d()
        self.drop5 = nn.Dropout2d()
        self.drop6 = nn.Dropout2d()

        out_channels = self.conv5.out_channels
        out_frames, out_features = self._calc_out_size((n_frames, n_features), [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], [self.pool1, self.pool2])

        flatten_size = out_channels * out_frames * out_features

        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256, self.output_shape[0] * self.output_shape[1])

    def _calc_out_size(self, in_size, convs:List[nn.Conv2d], pools:List[nn.MaxPool2d]):
        x, y = in_size
        if len(convs) and len(pools):
            conv, pool = convs[0], pools[0]
            x = (x - conv.kernel_size[0] + 1) // pool.kernel_size[0] // conv.stride[0]
            y = (y - conv.kernel_size[1] + 1) // pool.kernel_size[1] // conv.stride[1]
            return self._calc_out_size((x, y), convs[1:], pools[1:])
        if len(convs):
            conv = convs[0]
            x = (x - conv.kernel_size[0] + 1) // conv.stride[0]
            y = (y - conv.kernel_size[1] + 1) // conv.stride[1]
            return self._calc_out_size((x, y), convs[1:], pools[1:])
        if len(pools):
            pool = pools[0]
            x = x // pool.kernel_size[0]
            y = y // pool.kernel_size[1]
            return self._calc_out_size((x, y), convs[1:], pools[1:])
        return x, y

    def forward(self, X: torch.Tensor):
        X = X.reshape(-1, self.n_channels, self.n_frames, self.n_features)
        
        X = self.conv1(X)
        X = torch.relu(X)
        X = self.pool1(X)
        X = self.drop1(X)
        X = self.bn1(X)

        X = self.conv2(X)
        X = torch.relu(X)
        X = self.pool2(X)
        X = self.drop2(X)
        X = self.bn2(X)
        
        X = self.conv3(X)
        X = torch.relu(X)
        X = self.drop3(X)
        X = self.bn3(X)

        X = self.conv4(X)
        X = torch.relu(X)
        X = self.drop4(X)
        X = self.bn4(X)

        X = self.conv5(X)
        X = torch.relu(X)
        X = self.drop5(X)
        X = self.bn5(X)

        X = torch.flatten(X, start_dim=1)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.drop6(X)
        X = self.bn6(X)
        X = self.fc3(X)
        X = torch.sigmoid(X)
        
        outputs = X.reshape(-1, *self.output_shape)
        return outputs

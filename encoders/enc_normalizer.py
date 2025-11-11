import torch

from torch import nn, FloatTensor


class EncNormalizer(nn.Module):
    def __init__(self, enc_mean_path: str, enc_std_path: str, enc_max_path: str, enc_min_path: str):
        super().__init__()
        self.enc_mean = nn.Parameter(
            torch.load(enc_mean_path, map_location='cuda')[None, None, :],
            requires_grad=False
        )
        temp = torch.load(enc_std_path, map_location='cuda')[None, None, :]
        temp += torch.logical_and(abs(temp)<=0.01, (temp)>=0)*0.01
        temp -= torch.logical_and(abs(temp)<=0.01, (temp)<0)*0.01

        self.enc_std = nn.Parameter(
            temp, 
            #torch.load(enc_std_path, map_location='cuda')[None, None, :],
            requires_grad=False
        )

        self.enc_max = nn.Parameter(
            torch.load(enc_max_path, map_location='cuda')[None, None, :],
            requires_grad=False
        )

        self.enc_min = nn.Parameter(
            torch.load(enc_min_path, map_location='cuda')[None, None, :],
            requires_grad=False
        )
        self.epsilon = 1e-8
        self.scaled_maxv = 1.
        self.scaled_minv = -1.

    def forward(self, *args, **kwargs):
        return nn.Identity()(*args, **kwargs)

    def normalize(self, encoding: FloatTensor) -> FloatTensor:
        return (encoding - self.enc_mean) / self.enc_std

    def denormalize(self, pred_x_0: FloatTensor) -> FloatTensor:
        return pred_x_0 * self.enc_std + self.enc_mean
    
    def minmax_scaling(self, encoding: FloatTensor) -> FloatTensor:
        X_std = (encoding - self.enc_min) / (self.enc_max - self.enc_min + self.epsilon)
        X_scaled = X_std * (self.scaled_maxv - self.scaled_minv) + self.scaled_minv
        return X_scaled

    def undo_minmax_scaling(self, pred_x_0: FloatTensor) -> FloatTensor:
        x_std = (pred_x_0 - self.scaled_minv) / (self.scaled_maxv - self.scaled_minv)
        x = x_std * (self.enc_max - self.enc_min + self.epsilon) + self.enc_min
        return x

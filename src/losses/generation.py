"""Module containing generation losses."""

from typing import List, Optional, Union

import torch
import torchvision
from torch import Tensor
from torchvision.transforms import functional as F


class VGG19(torch.nn.Module):
    """Implementation of a VGG19 network for the VGGLoss."""

    def __init__(self, requires_grad: bool = False) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, X: Tensor) -> List[Tensor]:
        """Forward implementation for the module."""
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class MSELoss(torch.nn.MSELoss):
    """
    Extends the MSE loss to permit automatic resizing of the inputs to match
    size.

    :param auto_resize: resize the inputs to match shapes.
    """

    def __init__(
        self,
        auto_resize: bool = True,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.auto_resize = auto_resize

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward implementation for the loss."""
        # pylint: disable=arguments-renamed
        if self.auto_resize:
            min_width = min(x.shape[-2], y.shape[-2])
            min_height = min(x.shape[-1], y.shape[-1])

            x = F.resize(x, [min_width, min_height])
            y = F.resize(y, [min_width, min_height])

        return super().forward(x, y)


class VGGLoss(torch.nn.Module):
    """
    Implementation of the VGG Perception loss.

    :param auto_resize: resize the inputs to match shapes.
    """

    def __init__(self, auto_resize: bool = True) -> None:
        super().__init__()
        self.vgg = VGG19().to("cpu")
        self.auto_resize = auto_resize
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x: Tensor, y: Tensor) -> Union[Tensor, float]:
        """Forward implementation for the loss."""
        if self.vgg.device != x.device:
            self.vgg.to(x.device)

        if self.auto_resize:
            min_width = min(x.shape[-2], y.shape[-2])
            min_height = min(x.shape[-1], y.shape[-1])

            x = F.resize(x, [min_width, min_height])
            y = F.resize(y, [min_width, min_height])

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i, _ in enumerate(x_vgg):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class MSEVGGLoss(torch.nn.Module):
    """Apply both the MSE and the VGG loss."""

    def __init__(self) -> None:
        super().__init__()
        self.perception_loss = VGGLoss()
        self.reconstruction_loss = MSELoss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward implementation for the loss."""
        loss_1 = self.perception_loss(x, y)
        loss_2 = self.reconstruction_loss(x, y)

        return loss_1 + loss_2

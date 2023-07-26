import torch
from torch import nn


class PositionalEncoding2D(nn.Module):
    """Concatenate normalized position coordinates to the input tensor.

    This module concatenates the X and Y coordinates of the input tensor as two
    additional channels to it. X and Y coordinates are normalized between -1 and 1 where
    (0, 0) is the center of the image.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_grid(self, height: int, width: int) -> torch.Tensor:
        """Get the X and Y coordinates for a given height and width.

        Args:
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            torch.Tensor: (2, H, W)
        """
        return torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, height),
                torch.linspace(-1, 1, width),
                indexing="ij",
            ),
            dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        grid = self.get_grid(height, width).to(x.device)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        x = torch.cat([x, grid], dim=1)
        return x

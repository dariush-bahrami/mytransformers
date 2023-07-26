from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
from torch import Tensor, nn

from ..buildingblocks.layers import EncoderLayer


@dataclass
class EmbeddingsClassifierConfig:
    number_of_classes: List[str]
    number_of_encoders: int
    embedding_dimension: int
    number_of_heads: int
    attention_dropout_p: float
    feedforward_dimension: int
    feedforward_dropout_p: float


class EmbeddingsClassifier(nn.Module):
    """A classifier that maps a sequence of embeddings to a sequence of classes.

    Args:
        config (SequenceClassifierConfig): The configuration of the model.
    """

    def __init__(self, config: EmbeddingsClassifierConfig) -> None:
        super().__init__()
        self.__config = config
        self.encoders = nn.Sequential(
            *[
                EncoderLayer(
                    config.embedding_dimension,
                    config.number_of_heads,
                    config.attention_dropout_p,
                    config.feedforward_dimension,
                    config.feedforward_dropout_p,
                )
                for _ in range(config.number_of_encoders)
            ]
        )
        self.classifier = nn.Linear(
            config.embedding_dimension,
            self.number_of_classes,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x (Tensor): (B, S, E)

        Returns:
            Tensor: (B, C)
        """
        x = self.encoders(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

    @property
    def config(self) -> EmbeddingsClassifierConfig:
        return self.__config

    @classmethod
    def from_config(
        cls,
        config: EmbeddingsClassifierConfig,
    ) -> "EmbeddingsClassifierConfig":
        config_dict = asdict(config)
        return cls(**config_dict)

    def save_to_file(self, path: Path) -> None:
        config_dict = asdict(self.config)
        torch.save({"state_dict": self.state_dict(), "config_dict": config_dict}, path)

    @classmethod
    def load_from_file(cls, path: Path) -> "EmbeddingsClassifierConfig":
        data = torch.load(path)
        config = EmbeddingsClassifierConfig(**data["config_dict"])
        model = cls.from_config(config)
        model.load_state_dict(data["state_dict"])
        return model

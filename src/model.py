import torch
import torch.nn as nn
import torch.nn.functional as F


class LectureSenseRanking(nn.Module):
    """
    LectureSense multimodal ranking model.

    For every transcript sentence and candidate slide,
    the model produces a compatibility score.

    During training, the correct slide is encouraged
    to receive the highest score among all candidate slides.
    """

    def __init__(
        self,
        text_dim=384,
        image_dim=512,
        hidden_dim=256
    ):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Fusion:
        # text
        # image
        # |text - image|
        # text * image
        # cosine similarity
        fusion_dim = hidden_dim * 4 + 1

        self.scorer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 1)
        )

    def forward(
        self,
        text_emb,
        image_emb
    ):
        """
        Parameters
        ----------
        text_emb : torch.Tensor
            Shape [N, 384]

        image_emb : torch.Tensor
            Shape [N, 512]

        Returns
        -------
        torch.Tensor
            Compatibility scores, shape [N]
        """

        # Project both modalities into the same space.
        t = self.text_proj(text_emb)
        i = self.image_proj(image_emb)

        # Normalized representations.
        t_norm = F.normalize(
            t,
            p=2,
            dim=1
        )

        i_norm = F.normalize(
            i,
            p=2,
            dim=1
        )

        # Semantic similarity between text and image.
        cosine = torch.sum(
            t_norm * i_norm,
            dim=1,
            keepdim=True
        )

        # Multimodal interaction features.
        fusion = torch.cat(
            [
                t,
                i,
                torch.abs(t - i),
                t * i,
                cosine
            ],
            dim=1
        )

        # Compatibility score.
        score = self.scorer(
            fusion
        ).squeeze(1)

        return score
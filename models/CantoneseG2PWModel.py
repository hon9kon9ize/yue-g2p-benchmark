import sys
import os
from typing import List

sys.path.append("g2pW-Cantonese")

from models.G2PModel import G2PModel
from huggingface_hub import snapshot_download
from g2pw import G2PWConverter
import logging


logger = logging.getLogger(__name__)


class CantoneseG2PWModel(G2PModel):
    def __init__(
        self,
        g2pw_repo_id: str = "Naozumi0512/g2pW-Cantonese",
        bert_repo_id: str = "hon9kon9ize/bert-large-cantonese",
        batch_size=None,
        num_workers=8,
    ):
        # check if weights file exists
        if not os.path.exists("g2pW-Cantonese/checkpoints/g2pW-Cantonese"):
            logger.info("Downloading g2pw model...")
            snapshot_download(
                repo_id=g2pw_repo_id,
                local_dir="g2pW-Cantonese/checkpoints/g2pW-Cantonese",
            )

        if not os.path.exists("g2pW-Cantonese/checkpoints/bert-large-cantonese"):
            logger.info("Downloading BERT model...")
            snapshot_download(
                repo_id=bert_repo_id,
                local_dir="g2pW-Cantonese/checkpoints/bert-large-cantonese",
            )

        self.model = G2PWConverter(
            model_dir="g2pW-Cantonese/checkpoints/g2pW-Cantonese",
            model_source="g2pW-Cantonese/checkpoints/bert-large-cantonese",
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def get_name(self) -> str:
        return "g2pW-Cantonese"

    def _predict(self, texts: List[str]) -> List[str]:
        preds = self.model(texts)
        results = []

        for pred in preds:
            results.append(" ".join([t for t in pred if t is not None]))

        return results

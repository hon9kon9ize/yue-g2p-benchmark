from typing import List
from models.G2PModel import G2PModel
import ToJyutping


class ToJyutpingModel(G2PModel):
    def get_name(self) -> str:
        return "ToJyutping"

    def _predict(self, texts: List[str]) -> List[str]:
        return [ToJyutping.get_jyutping_text(text) for text in texts]

from typing import List, Union
from models.G2PModel import G2PModel
import ToJyutping


class ToJyutpingModel(G2PModel):
    def get_name(self) -> str:
        return "ToJyutping"

    def _predict(self, texts: List[str]) -> List[List[Union[str, None]]]:
        return [[result[1] for result in ToJyutping.get_jyutping_list(text)] for text in texts]

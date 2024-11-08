from typing import List
from models.G2PModel import G2PModel
import re
import pycantonese


class PyCantoneseModel(G2PModel):
    def get_name(self) -> str:
        return "PyCantonese"

    def _predict(self, texts: List[str]) -> List[str]:
        predictions = []

        for text in texts:
            pred = " ".join(
                [
                    k
                    for j in pycantonese.characters_to_jyutping(text)
                    if j[1] != None
                    for k in re.split(r"([a-z]+[1-6])", j[1])
                ]
            )
            pred = re.sub(r"\s+", " ", pred)
            predictions.append(pred)

        return predictions

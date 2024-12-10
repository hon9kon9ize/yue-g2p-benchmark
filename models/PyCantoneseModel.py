from typing import List, Union
from models.G2PModel import G2PModel
import re
import pycantonese


class PyCantoneseModel(G2PModel):
    def get_name(self) -> str:
        return "PyCantonese"

    def _predict(self, texts: List[str]) -> List[List[Union[str, None]]]:
        return [
            [
                jyutping
                for result in pycantonese.characters_to_jyutping(text)
                for jyutping in result[1] and re.findall(r"[a-z]+[1-6]", result[1]) or [None]
            ]
            for text in texts
        ]

import re
import json
from typing import Dict, List
from models.G2PModel import G2PModel
from tqdm.auto import tqdm
import requests
from urllib.parse import urlencode


class GoogleTranslateModel(G2PModel):
    def get_name(self) -> str:
        return "GoogleTranslate"

    def __init__(self, proxies: Dict = None):
        self.proxies = proxies

    def _translate(self, text: str) -> str:
        url = "https://translate.google.com/_/TranslateWebserverUi/data/batchexecute"
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,zh-TW;q=0.6",
            "cache-control": "no-cache",
            "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
            "origin": "https://translate.google.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://translate.google.com/",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        }
        param = json.dumps([[text, "yue", "yue", True], [1]])
        freq = json.dumps([[["MkEWBc", param, None, "generic"]]])
        freq = {"f.req": freq}
        freq = urlencode(freq)

        response = requests.post(
            url,
            headers=headers,
            data=freq,
            proxies=self.proxies,
            timeout=10,
        )

        try:
            jyutping = json.loads(json.loads(response.text.split("\n")[2])[0][2])[0][0]

            if jyutping is None:
                return ""

            pred = " ".join(
                [k for j in jyutping.split(" ") for k in re.split(r"([a-z]+[1-6])", j)]
            )
            pred = re.sub(r"\s+", " ", pred)

            return pred
        except json.JSONDecodeError:
            return ""

    def _predict(self, texts: List[str]) -> List[str]:
        results = []

        for text in tqdm(texts):
            results.append(self._translate(text))

        return results

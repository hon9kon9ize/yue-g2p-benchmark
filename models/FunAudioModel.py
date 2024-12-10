import os
from typing import List, Union
import logging
from modelscope import snapshot_download
from models import G2PModel

logger = logging.getLogger(__name__)


class FunAudioModel(G2PModel):
    """
    FunAudioModel is a wrapper for the FunAudio's ttsfrd model. ttsfrd is a TTS frontend engine used to convert text to phonemes. ttsfrd is a closed-sourced library.
    """

    def __init__(self, repo_id="iic/CosyVoice-ttsfrd"):
        if not os.path.exists("pretrained_models/CosyVoice-ttsfrd"):
            logger.info("Downloading FunAudio ttsfrd model...")

            # git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
            # cd pretrained_models/CosyVoice-ttsfrd
            # unzip resource.zip -d .
            # pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl

            snapshot_download(repo_id, local_dir="pretrained_models/CosyVoice-ttsfrd")

            os.system(
                "cd pretrained_models/CosyVoice-ttsfrd && unzip resource.zip -d ."
            )
            os.system(
                "pip install pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl"
            )

        import ttsfrd  # type: ignore

        self.model = ttsfrd.TtsFrontendEngine()
        self.model.initialize("pretrained_models/CosyVoice-ttsfrd/resource")
        self.model.set_lang_type("hk")

    def get_name(self) -> str:
        return "FunAudio"

    def _predict(self, texts: List[str]) -> List[List[Union[str, None]]]:
        predictions = []

        for text in texts:
            try:
                model_output = self.model.gen_tacotron_symbols(text)
                phonemes = (
                    phoneme.replace("{", "").replace("}", "").split("$")
                    for sentence in model_output.strip().split("\n")
                    for phoneme in sentence.split("\t")[1].strip().split(" ")
                )
                pred = []
                curr_syllable = ""

                for s in phonemes:
                    if s[0].startswith("#") and s[0] != "#1":
                        pred.append(None)
                        continue

                    syllable = s[0].split("_")[0]
                    if s[2] == "s_end":
                        tone = s[1][-1]
                        pred.append(curr_syllable + syllable + tone)
                        curr_syllable = ""
                    elif s[2] != "s_none":
                        curr_syllable += syllable

                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error: {e}")
                predictions.append([])

        return predictions

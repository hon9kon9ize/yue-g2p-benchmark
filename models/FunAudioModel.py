import os
from typing import List
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

    def _predict(self, texts: List[str]) -> List[str]:
        predictions = []

        for text in texts:
            try:
                model_output = (
                    self.model.gen_tacotron_symbols(text).strip().split("\t")[1]
                )
                syllables = [
                    s.replace("{", "").replace("}", "").split("$")
                    for s in model_output.split(" ")
                ]
                pred = ""

                while True:
                    try:
                        s = syllables.pop(0)
                        syllable = s[0].split("_")[0]
                        tone = s[1][-1]
                        if s[2] == "s_end":
                            pred += syllable + tone + " "
                        elif s[2] != "s_none":
                            pred += syllable

                    except IndexError:
                        break

                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error: {e}")
                predictions.append("")
                continue

        return predictions

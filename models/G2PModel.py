from typing import List, Union
import re


class G2PModel:
    """
    Abstract base class for G2P (Grapheme-to-Phoneme) models.

    This class provides a template for G2P models with methods for predicting
    phonemes from text and cleaning Jyutping strings.
    """

    def _predict(self, texts: List[str]) -> List[List[Union[str, None]]]:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def _clean_jyutpings(self, jyutpings: List[Union[str, None]]) -> List[Union[str, None]]:
        """
        Cleans up a list of Jyutping strings by removing any non-Jyutping text.

        A Jyutping string consists of syllables that match the pattern of one or more
        lowercase letters followed by a digit from 1 to 6. This function filters out
        any syllables that do not match this pattern.

        Args:
            jyutpings (List[str]): A list of Jyutping strings to be cleaned.

        Returns:
            List[str]: A list of cleaned Jyutping strings with only valid Jyutping syllables.
        """
        return [jyutping if jyutping and re.fullmatch(r"[a-z]+[1-6]", jyutping) else None for jyutping in jyutpings]

    def __call__(self, texts: List[str]) -> List[List[Union[str, None]]]:
        return [self._clean_jyutpings(result) for result in self._predict(texts)]

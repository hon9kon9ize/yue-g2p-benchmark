from typing import List
import re


class G2PModel:
    """
    Abstract base class for G2P (Grapheme-to-Phoneme) models.

    This class provides a template for G2P models with methods for predicting
    phonemes from text and cleaning Jyutping strings.
    """

    def _predict(self, texts: List[str]) -> List[str]:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def _clean_jyutpings(self, jyutpings: List[str]) -> List[str]:
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
        cleaned_jyutpings = []

        for jyutping in jyutpings:
            cleaned_jyutpings.append(" ".join(re.findall(r"[a-z]+[1-6]", jyutping)))

        return cleaned_jyutpings

    def __call__(self, texts: List[str]) -> List[str]:
        results = self._predict(texts)

        return self._clean_jyutpings(results)

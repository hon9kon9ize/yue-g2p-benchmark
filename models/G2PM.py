import torch
import pickle
import os
from typing import List
from models.G2PModel import G2PModel
from g2pM.g2pM2 import G2pM
import sys
from tqdm import tqdm

sys.path.append("g2pM")


class G2PM_Model(G2PModel):
    def __init__(self):
        super().__init__()
        self.model = G2pM()

    def load_variable(self, state_dict):
        self.model.load_state_dict(state_dict)

    def load_trained_model(self, ckpt_file):
        state_dict = pickle.load(open(ckpt_file, "rb"))
        self.load_variable(state_dict)

    def _predict(self, texts: List[str]) -> List[str]:
        predictions = [self.model(text, tone=True, char_split=False) for text in texts]
        # Convert list of lists to a single string for each prediction
        return [" ".join(prediction) for prediction in predictions]

    def get_name(self):
        return "G2PM_Model"

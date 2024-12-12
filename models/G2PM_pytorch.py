import torch
import pickle
from typing import List
from models.G2PModel import G2PModel
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import sys
import os

sys.path.append("g2pM")


class G2pM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, padding_idx):
        super(G2pM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Bidirectional
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, inputs, target_indices):
        """
        Args:
            inputs: [batch_size, seq_len]
            target_indices: list of lists containing target positions for each sample
        Returns:
            logits: [total_targets, num_classes]
        """
        embedded = self.embedding(inputs)  # [batch_size, seq_len, embed_dim]
        packed_output, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]

        # Extract target hidden states
        target_hidden = []
        for i, indices in enumerate(target_indices):
            for idx in indices:
                if idx < packed_output.size(
                    1
                ):  # Ensure index is within sequence length
                    target_hidden.append(packed_output[i, idx, :])
        if target_hidden:
            target_hidden = torch.stack(target_hidden)  # [total_targets, hidden_dim*2]
        else:
            target_hidden = torch.empty(0, self.lstm.hidden_size * 2).to(
                packed_output.device
            )

        logits = self.fc(target_hidden)  # [total_targets, num_classes]
        return logits


class G2PM_pytorch_Model(G2PModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.char2idx = None
        self.idx2class = None
        # Use relative path to access the model file in the g2pM directory
        self.load_trained_model("g2pM/trained_pytorch_final.pth")

    def load_trained_model(self, ckpt_file):
        # Load mappings
        with open("g2pM/char2idx.pkl", "rb") as f:
            self.char2idx = pickle.load(f)
        with open("g2pM/class2idx.pkl", "rb") as f:
            class2idx = pickle.load(f)
        self.idx2class = {idx: cls for cls, idx in class2idx.items()}

        # Parameters
        vocab_size = len(self.char2idx)
        embed_dim = 128
        hidden_dim = 256
        num_classes = len(class2idx)
        pad_idx = self.char2idx["<PAD>"]

        # Initialize model
        self.model = G2pM(
            vocab_size, embed_dim, hidden_dim, num_classes, padding_idx=pad_idx
        )
        self.model.to(self.device)

        # Load the trained model
        self.model.load_state_dict(torch.load(ckpt_file, map_location=self.device, weights_only=True))
        self.model.eval()  # Set model to evaluation mode

    def _predict(self, texts: List[str]) -> List[str]:
        predictions = []
        for text in texts:
            predicted_labels = self.evaluate_sentence(text)
            predictions.append(" ".join(predicted_labels))
        return predictions

    def evaluate_sentence(
        self, sentence: str, pad_token="<PAD>", unk_token="<UNK>"
    ) -> List[str]:
        self.model.eval()
        with torch.no_grad():
            # Convert sentence to indices
            input_ids = [
                self.char2idx.get(char, self.char2idx[unk_token]) for char in sentence
            ]
            input_tensor = (
                torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            )  # [1, seq_len]

            # Since it's a single sentence, target_indices are all positions (or specific based on your use case)
            # Assuming you want predictions for all characters
            target_indices = [list(range(len(input_ids)))]

            # Get logits
            logits = self.model(input_tensor, target_indices)  # [seq_len, num_classes]
            if logits.numel() == 0:
                print("No target indices found in the sentence.")
                return []

            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()  # [seq_len]

            # Map predictions to class labels
            predicted_labels = [self.idx2class.get(idx, "<UNK>") for idx in predictions]

        return predicted_labels

    def get_name(self):
        return "G2PM_pytorch_Model"

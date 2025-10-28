import json, torch
from torch.utils.data import Dataset

class ARCDataset(Dataset):
    def __init__(self, path="training.json", split="train"):
        with open(path, "r") as f:
            self.data = json.load(f)
        self.items = []
        for prob_id, ex in self.data.items():
            for i, samp in enumerate(ex[split]):
                self.items.append({
                    "prob_id": prob_id,
                    "idx": i,
                    "input": torch.tensor(samp["input"], dtype=torch.long),
                    "output": torch.tensor(samp["output"], dtype=torch.long) if samp["output"] else None
                })
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

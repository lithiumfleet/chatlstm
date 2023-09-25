from tokenutil import tokenizer
from torch.utils.data import Dataset, DataLoader

def TextDataLoader():
    ds = TextDataset('./dataset/dialog_clean3_20.txt')
    return DataLoader(
        dataset=ds,
        batch_size=32,
        shuffle=False,
        num_workers=6
    )

class TextDataset(Dataset):
    def __init__(self, ds_pth:str) -> None:
        self.ds = self._get_ds(ds_pth)

    def _get_ds(self, ds_pth) -> list:
        texts = []
        with open(ds_pth, 'r') as f:
            for line in f.readlines():
                texts.append(line.rstrip())
        return texts

    def __len__(self):
        """
        assuming that sample(x,y) could be any two neighbors, 
        because the chat dataset using long terms dialogs.
        """
        return len(self.ds)-1

    def __getitem__(self, i):
        x = tokenizer(self.ds[i])
        y = tokenizer(self.ds[i+1])
        return (x,y)
        
if __name__ == '__main__':
    from tokenutil import vec2str
    ds = TextDataset('./dataset/dialog_clean2.txt')
    print()
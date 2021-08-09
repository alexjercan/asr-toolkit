import os
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .general import pad_last


class LibriSpeechDataset(Dataset):
    def __init__(self, dictionary, root, urls, folder_in_archive="LibriSpeech", download=False, transform=None):
        super().__init__()
        datasets = [torchaudio.datasets.LIBRISPEECH(root, url, folder_in_archive, download) for url in urls]

        self.dataset = ConcatDataset(datasets)
        self.transform = transform
        self.dictionary = dictionary

    def __len__(self):
        return  len(self.dataset)

    def __getitem__(self, n):
        waveform, _, utterance, _, _, _ = self.dataset[n]

        if self.transform is not None:
            waveform = self.transform(waveform)

        wlen = waveform.shape[-1]
        ulen = len(utterance)

        utterance = torch.tensor(list(map(lambda c: self.dictionary[c], utterance.upper())))
        return wlen, waveform[0], ulen, utterance


def collate_fn(batch):
    wlen, waveform, ulen, utterance = zip(*batch)
    waveform = pad_last(waveform)
    return torch.tensor(wlen), torch.stack(waveform, 0), torch.tensor(ulen), torch.cat(utterance)


def librispeech_dataloader(dictionary, root, urls, folder_in_archive="LibriSpeech", download=False, transform=None, batch_size=2, workers=8, pin_memory=True, shuffle=False):
    dataset = LibriSpeechDataset(dictionary, root, urls, folder_in_archive, download, transform)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=pin_memory, shuffle=shuffle, collate_fn=collate_fn)
    return dataset, dataloader
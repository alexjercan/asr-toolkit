import os
import json
import torch
import urllib.request
import torchaudio.datasets
import torch.nn.functional as F

from torch.utils.data import Dataset, ConcatDataset, DataLoader


vocab_url = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json"

with urllib.request.urlopen(vocab_url) as url:
    data = json.loads(url.read().decode())

DICTIONARY = data
LABELS = {v: k for k, v in DICTIONARY.items()}


def tensor_to_string(tensor, target_lengths):
    result = []
    s = 0
    for i in range(target_lengths.shape[0]):
        result.append("".join([LABELS[t.item()]
                      for t in tensor[s:s+target_lengths[i]]]))
        s = s + target_lengths[i]
    return result


class LibriSpeechDataset(Dataset):
    def __init__(self, root, urls, folder_in_archive="LibriSpeech", download=False, wave_transform=None):
        super().__init__()
        datasets = [torchaudio.datasets.LIBRISPEECH(
            root, url, folder_in_archive, download) for url in urls]

        self.dataset = ConcatDataset(datasets)
        self.wave_transform = wave_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, n):
        waveform, _, utterance, _, _, _ = self.dataset[n]

        if self.wave_transform is not None:
            waveform = self.wave_transform(waveform)

        wlen = waveform.shape[-1]
        ulen = len(utterance)

        utterance = "".join(list(utterance.replace(" ", "|")))
        utterance = torch.tensor(
            list(map(lambda c: DICTIONARY[c], utterance.upper())))
        return wlen, waveform[0], ulen, utterance


def pad_last(tensor):
    maxlen = max(map(lambda t: t.shape[-1], tensor))
    return list(map(lambda t: F.pad(t, (0, maxlen - t.shape[-1])), tensor))


def collate_fn(batch):
    wlen, waveform, ulen, utterance = zip(*batch)
    waveform = pad_last(waveform)
    return torch.tensor(wlen), torch.stack(waveform, 0), torch.tensor(ulen), torch.cat(utterance)


def librispeech_dataloader(root, urls, folder_in_archive="LibriSpeech", download=False, wave_transform=None, batch_size=2, workers=8, pin_memory=True, shuffle=False):
    dataset = LibriSpeechDataset(
        root, urls, folder_in_archive, download, wave_transform)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw,
                            pin_memory=pin_memory, shuffle=shuffle, collate_fn=collate_fn)
    return dataset, dataloader

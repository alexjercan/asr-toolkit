import os
import torch
import torchaudio

import pandas as pd
from pathlib import Path
from torchaudio.datasets.librispeech import load_librispeech_item

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


class LibriSpeechBookDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, root, url, folder_in_archive="LibriSpeech", download=False):
        super(LibriSpeechBookDataset, self).__init__(root, url, folder_in_archive, download)

        chapterpaths = {p.stem:str(p) for p in Path(self._path).glob('*/*/')}
        names = ["ID", "READER", "MINUTES", "SUBSET", "PROJ.", "BOOK ID", "CH. TITLE", "PROJECT TITLE"]
        converters = {"BOOK ID": str.strip, "SUBSET": str.strip, "CH. TITLE" : str.strip, "PROJECT TITLE" : str.strip}
        chapters = os.path.join(root, folder_in_archive, "CHAPTERS.TXT")
        df = pd.read_csv(chapters, delimiter='|', comment=';', names=names, converters=converters)
        df = df[df["SUBSET"] == os.path.basename(url)].groupby("BOOK ID")
        df = pd.DataFrame({"CHAPTERS": df["ID"].apply(list), "MINUTES": df["MINUTES"].apply(sum)})
        df['CHAPTER_PATH'] = df.apply(lambda row: [chapterpaths[str(x)] for x in row["CHAPTERS"]], axis=1)
        df.reset_index(level=0, inplace=True)
        df = df.astype({"BOOK ID": 'object'})

        names = ["BOOK ID", "BOOK TITLE"]
        converters = {"BOOK ID": str.strip, "BOOK TITLE": str.strip}
        books = os.path.join(root, folder_in_archive, "BOOKS.TXT")
        dfp = pd.read_csv(books, delimiter='|', comment=';', names=names, converters=converters, usecols=names, lineterminator='\n')
        df = pd.merge(df, dfp, how="inner", on="BOOK ID")

        self._walker = df

    def __getitem__(self, n):
        row = self._walker.iloc[n]

        audiofileids = [str(p.stem) for chapterpath in row["CHAPTER_PATH"] for p in Path(chapterpath).glob('*' + self._ext_audio)]
        items = [load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt) for fileid in audiofileids]

        waveforms, _, utterances, _, _, _ = zip(*items)
        return torch.cat(waveforms, dim=1), " ".join(utterances), row["BOOK TITLE"], row["MINUTES"]
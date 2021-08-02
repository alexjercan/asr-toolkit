import numpy as np
import torch.nn as nn


class CTCLossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss = nn.CTCLoss()

        self.total_size = 0
        self.loss_sum = {}
        self.loss_avg = {}

    def forward(self, predictions, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc_loss(predictions, targets,
                                 input_lengths, target_lengths)

        loss_val = {}
        loss_val["CTC"] = ctc_loss.item()
        self.total_size += 1

        self.loss_avg = avg_error(self.loss_sum, loss_val, self.total_size)

        return ctc_loss

    def show(self):
        ctc = self.loss_avg["CTC"]
        return f"(ctc:{ctc:.4f})"


class ASRMetricFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_size = 0
        self.error_sum = {}
        self.error_avg = {}

    def forward(self, predictions, targets):
        error_val = {}
        batch_size = len(predictions)

        error_val["WER"] = sum(
            map(get_word_error_rate, zip(predictions, targets)))
        error_val["CER"] = sum(
            map(get_char_error_rate, zip(predictions, targets)))
        self.total_size += batch_size

        self.error_avg = avg_error(self.error_sum, error_val, self.total_size)
        return self.error_avg

    def show(self):
        error = self.error_avg
        format_str = ("\n======WAV2VEC2========\nWER=%.4f\tCER=%.4f\n")
        return format_str % (error["WER"], error["CER"])


def get_char_error_rate(pairs):
    target, changed = pairs
    r = list(target)
    h = list(changed)
    d = get_sentence_error(r, h)
    return float(d[len(r)][len(h)]) / len(r) * 100


def get_word_error_rate(pairs):
    target, changed = pairs
    r = target.split("|")
    h = changed.split(" ")
    d = get_sentence_error(r, h)
    return float(d[len(r)][len(h)]) / len(r) * 100


# Reference: https://github.com/imalic3/python-word-error-rate
def get_sentence_error(reference, hypothesis):
    r = reference
    h = hypothesis
    
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d


def avg_error(error_sum, error_val, total_size):
    error_avg = {}
    for item, value in error_val.items():
        error_sum[item] = error_sum.get(item, 0) + value
        error_avg[item] = error_sum[item] / float(total_size)
    return error_avg

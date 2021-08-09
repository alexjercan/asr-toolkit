import torch
import torch.nn as nn

from tqdm import tqdm

from .general import tensors_to_device, tensor_to_string
from .metrics import CTCLossFunction


class GreedyDecoder(nn.Module):
    def __init__(self, labels_map, blank_id, batch_dim_index=0):
        super().__init__()
        self.labels_map = labels_map
        self.blank_id = blank_id
        self.batch_dim_index = batch_dim_index

    def forward(self, predictions: torch.Tensor, predictions_len: torch.Tensor = None):
        hypotheses = []
        prediction_cpu_tensor = predictions.long().cpu()
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]
            decoded_prediction = []
            previous = self.blank_id
            for p in prediction:
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                previous = p

            text = self.decode_tokens_to_str(decoded_prediction)

            hypothesis = text

            hypotheses.append(hypothesis)
        return hypotheses

    def decode_tokens_to_str(self, tokens):
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis

    def decode_ids_to_tokens(self, tokens):
        token_list = [self.labels_map[c] for c in tokens if c != self.blank_id]
        return token_list


class QuartzNet(nn.Module):
    def __init__(self, nemo_model, labels, blank):
        super(QuartzNet, self).__init__()

        self.preprocessor = nemo_model.preprocessor
        self.encoder = nemo_model.encoder
        self.decoder = nemo_model.decoder

        self.ctc_decoder_predictions_tensor = GreedyDecoder(labels, blank)

        self.labels = labels
        self.blank = blank

    def forward(self, input_signal, input_signal_length=None):
        processed_signal, processed_signal_length = self.preprocessor(input_signal=input_signal, length=input_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions

    def train_model(self, train_dataloader, val_dataloader, metric_fn, optimizer, scaler, epoch_idx):
        self.train()
        device = next(self.parameters()).device

        loss_fn = CTCLossFunction(blank=self.blank)
        loop = tqdm(train_dataloader, position=0, leave=True)

        for _, tensors in enumerate(loop):
            valid_lengths, waveform, target_lengths, utterance = tensors_to_device(tensors, device)

            log_probs, encoded_len, _ = self.forward(waveform, valid_lengths)
            encoded_len = encoded_len.clamp_max(log_probs.shape[1])
            loss = loss_fn(log_probs.permute(1, 0, 2), utterance, encoded_len, target_lengths)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss_fn.show(), epoch=epoch_idx)
        loop.close()
        self.test_model(val_dataloader, metric_fn)

    def test_model(self, dataloader, metric_fn):
        self.eval()
        device = next(self.parameters()).device

        loss_fn = CTCLossFunction(blank=self.blank)
        loop = tqdm(dataloader, position=0, leave=True)

        for _, tensors in enumerate(loop):
            valid_lengths, waveform, target_lengths, utterance = tensors_to_device(tensors, device)

            with torch.no_grad():
                log_probs, encoded_len, greedy_predictions = self.forward(waveform, valid_lengths)
                loss_fn(log_probs.permute(1, 0, 2), utterance, encoded_len, target_lengths)

                transcriptions = self.ctc_decoder_predictions_tensor(greedy_predictions, predictions_len=encoded_len)

            metric_fn(tensor_to_string(utterance, target_lengths, self.labels), transcriptions)

            loop.set_postfix(loss=loss_fn.show())
        loop.close()

    def inference(self, input_value, input_value_length):
        self.eval()

        with torch.no_grad():
            _, encoded_len, greedy_predictions = self.forward(input_value, input_value_length)
            transcriptions = self.ctc_decoder_predictions_tensor(greedy_predictions, predictions_len=encoded_len)

        return transcriptions
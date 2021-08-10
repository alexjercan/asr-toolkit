import torch
import torch.nn as nn


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
    def __init__(self, nemo_model, decoder):
        super(QuartzNet, self).__init__()

        self.preprocessor = nemo_model.preprocessor
        self.encoder = nemo_model.encoder
        self.decoder = nemo_model.decoder

        self.ctc_decoder_predictions_tensor = decoder

    def forward(self, input_signal, input_signal_length=None):
        processed_signal, processed_signal_length = self.preprocessor(input_signal=input_signal, length=input_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions

    def inference(self, input_value, input_value_length):
        self.eval()

        with torch.no_grad():
            _, encoded_len, greedy_predictions = self.forward(input_value, input_value_length)
            transcriptions = self.ctc_decoder_predictions_tensor(greedy_predictions, predictions_len=encoded_len)

        return transcriptions


class Wav2Vec2(nn.Module):
    def __init__(self, model, tokenizer):
        super(Wav2Vec2, self).__init__()

        self.model = model
        self.tokenizer = tokenizer

    def forward(self, waveform, valid_lengths=None):
        return self.model(waveform, valid_lengths)

    def decode(self, prediction):
        return self.tokenizer.decode(prediction)

    def inference(self, input_value):
        self.eval()

        with torch.no_grad():
            logits, _ = self.forward(input_value)
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.decode(predicted_ids[0])

        return transcription

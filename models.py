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


# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class BeamSearchDecoderWithLM(nn.Module):
    """Neural Module that does CTC beam search with a N-gram language model.
    It takes a batch of log_probabilities. Note the bigger the batch, the
    better as processing is parallelized. Outputs a list of size batch_size.
    Each element in the list is a list of size beam_search, and each element
    in that list is a tuple of (final_log_prob, hyp_string).
    Args:
        vocab (list): List of characters that can be output by the ASR model. For English, this is the 28 character set
            {a-z '}. The CTC blank symbol is automatically added.
        beam_width (int): Size of beams to keep and expand upon. Larger beams result in more accurate but slower
            predictions
        alpha (float): The amount of importance to place on the N-gram language model. Larger alpha means more
            importance on the LM and less importance on the acoustic model.
        beta (float): A penalty term given to longer word sequences. Larger beta will result in shorter sequences.
        lm_path (str): Path to N-gram language model
        num_cpus (int): Number of CPUs to use
        cutoff_prob (float): Cutoff probability in vocabulary pruning, default 1.0, no pruning
        cutoff_top_n (int): Cutoff number in pruning, only top cutoff_top_n characters with highest probs in
            vocabulary will be used in beam search, default 40.
    """

    def __init__(
        self, vocab, beam_width, alpha, beta, lm_path, num_cpus, cutoff_prob=1.0, cutoff_top_n=40
    ):

        try:
            from ctc_decoders import Scorer, ctc_beam_search_decoder_batch
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "BeamSearchDecoderWithLM requires the installation of ctc_decoders "
                "from scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh"
            )

        super(BeamSearchDecoderWithLM, self).__init__()

        if lm_path is not None:
            self.scorer = Scorer(alpha, beta, model_path=lm_path, vocabulary=vocab)
        else:
            self.scorer = None
        self.beam_search_func = ctc_beam_search_decoder_batch
        self.vocab = vocab
        self.beam_width = beam_width
        self.num_cpus = num_cpus
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n

    @torch.no_grad()
    def forward(self, log_probs, log_probs_length):
        probs = torch.exp(log_probs)
        probs_list = []
        for i, prob in enumerate(probs):
            probs_list.append(prob[: log_probs_length[i], :])

        res = self.beam_search_func(
            probs_list,
            self.vocab,
            beam_size=self.beam_width,
            num_processes=self.num_cpus,
            ext_scoring_func=self.scorer,
            cutoff_prob=self.cutoff_prob,
            cutoff_top_n=self.cutoff_top_n,
        )
        return res


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

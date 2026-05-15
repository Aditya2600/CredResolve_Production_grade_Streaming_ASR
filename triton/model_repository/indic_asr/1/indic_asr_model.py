"""Vendored fork of ai4bharat/indic-conformer-600m-multilingual `model_onnx.py`.

Phase 2 changes vs. upstream:
  * The encoder no longer runs as an in-process onnxruntime session. Instead,
    `encode()` issues a BLS call to the `indic_asr_encoder` Triton model, which
    serves `encoder.onnx` via the onnxruntime backend (GPU). This unifies the
    encoder execution path with the CTC ensemble (`indic_asr_ctc`) and lets
    Triton manage encoder placement, batching and TRT acceleration.
  * The preprocessor is pinned to CPU here (instead of the upstream
    `cuda if available` path) so the TorchScript preprocessor never races with
    the GPU encoder for device memory.

Everything else — RNNT greedy loop, joint network calls, CTC decode, timestamp
computation, vocab/language_mask handling — is preserved verbatim from upstream
so behavior matches the HF reference implementation.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort
import torch
import triton_python_backend_utils as pb_utils
from transformers import PretrainedConfig, PreTrainedModel


# ONNX components loaded in-process. The `encoder` component is intentionally
# absent: it now runs via BLS to `indic_asr_encoder`.
_INPROC_COMPONENT_NAMES = (
    ['ctc_decoder', 'rnnt_decoder', 'joint_enc', 'joint_pred', 'joint_pre_net']
    + [f'joint_post_net_{z}' for z in [
        'as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 'ks', 'mai', 'ml',
        'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur',
    ]]
)

_ENCODER_BLS_MODEL_NAME = 'indic_asr_encoder'
_ENCODER_MIN_FRAMES = 100


def _pb_tensor_to_numpy(tensor, name):
    if tensor is None:
        raise pb_utils.TritonModelException(f'Missing encoder output tensor: {name}')

    try:
        if tensor.is_cpu():
            return tensor.as_numpy()
    except AttributeError:
        return tensor.as_numpy()

    # TensorRT outputs can stay in GPU memory when returned through BLS.
    # The remaining RNNT/CTC decode path still uses in-process ORT sessions
    # with NumPy inputs, so copy to host at this boundary.
    try:
        torch_tensor = torch.utils.dlpack.from_dlpack(tensor.to_dlpack())
        return torch_tensor.detach().cpu().numpy()
    except Exception as exc:
        raise pb_utils.TritonModelException(
            f'Failed to copy GPU encoder output {name!r} to NumPy: {exc}'
        ) from exc


class IndicASRConfig(PretrainedConfig):
    model_type = "iasr"

    def __init__(self, ts_folder: str = "path", BLANK_ID: int = 256, RNNT_MAX_SYMBOLS: int = 10,
                 PRED_RNN_LAYERS: int = 2, PRED_RNN_HIDDEN_DIM: int = 640, SOS: int = 5632, **kwargs):
        super().__init__(**kwargs)
        self.ts_folder = ts_folder
        self.BLANK_ID = BLANK_ID
        self.RNNT_MAX_SYMBOLS = RNNT_MAX_SYMBOLS
        self.PRED_RNN_LAYERS = PRED_RNN_LAYERS
        self.PRED_RNN_HIDDEN_DIM = PRED_RNN_HIDDEN_DIM
        self.SOS = SOS

        if 'FRAME_DURATION_MS' not in kwargs:
            print('Please check FRAME_DURATION_MS. The timestamps can be inaccurate')
            fs = 0.08
        else:
            fs = kwargs['FRAME_DURATION_MS']

        self.FRAME_DURATION_MS = fs


class IndicASRModel(PreTrainedModel):
    config_class = IndicASRConfig

    def __init__(self, config):
        super().__init__(config)

        self.models = {}
        self.d = torch.device('cpu')
        self.models['preprocessor'] = torch.jit.load(
            f'{config.ts_folder}/assets/preprocessor.ts', map_location=self.d
        )

        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if torch.cuda.is_available() else ['CPUExecutionProvider']
        )
        for n in _INPROC_COMPONENT_NAMES:
            component_path = f'{config.ts_folder}/assets/{n}.onnx'
            if os.path.exists(component_path):
                self.models[n] = ort.InferenceSession(component_path, providers=providers)
            else:
                self.models[n] = None
                print('Failed to load', component_path)

        with open(f'{config.ts_folder}/assets/vocab.json') as reader:
            self.vocab = json.load(reader)

        with open(f'{config.ts_folder}/assets/language_masks.json') as reader:
            self.language_masks = json.load(reader)

    def forward(self, wav, lang, decoding='ctc', compute_timestamps=None, _timer=None):
        encoder_outputs, encoded_lengths = self.encode(wav, _timer=_timer)
        if _timer is not None:
            try:
                _timer.num_frames = int(np.asarray(encoded_lengths).reshape(-1)[0])
            except Exception:
                pass
        if decoding == 'ctc':
            return self._ctc_decode(encoder_outputs, encoded_lengths, lang, compute_timestamps, _timer=_timer)
        if decoding == 'rnnt':
            return self._rnnt_decode(encoder_outputs, encoded_lengths, lang, _timer=_timer)

    def encode(self, wav, _timer=None):
        # Stage 1: preprocessing (TorchScript log-mel filterbank, CPU)
        if _timer is not None:
            _timer.start('preproc')
        audio_signal, length = self.models['preprocessor'](
            input_signal=wav.to(self.d),
            length=torch.tensor([wav.shape[-1]]).to(self.d),
        )
        audio_signal_np = np.ascontiguousarray(audio_signal.cpu().numpy().astype(np.float32, copy=False))
        length_np = np.ascontiguousarray(length.cpu().numpy().astype(np.int64, copy=False))
        feature_frames = int(audio_signal_np.shape[-1])
        if feature_frames < _ENCODER_MIN_FRAMES:
            # TensorRT validates the physical input shape against the engine
            # profile (`minShapes=...x100`) before it looks at `length`.
            # Pad only the feature tensor; keep `length_np` untouched so decode
            # and timestamps still reflect the original utterance duration.
            audio_signal_np = np.ascontiguousarray(
                np.pad(
                    audio_signal_np,
                    ((0, 0), (0, 0), (0, _ENCODER_MIN_FRAMES - feature_frames)),
                    mode='constant',
                )
            )
        if _timer is not None:
            _timer.stop()

        # Stage 2: encoder forward (BLS into TRT/ORT-CUDA)
        if _timer is not None:
            _timer.start('encoder')
        encoder_request = pb_utils.InferenceRequest(
            model_name=_ENCODER_BLS_MODEL_NAME,
            requested_output_names=['outputs', 'encoded_lengths'],
            inputs=[
                pb_utils.Tensor('audio_signal', audio_signal_np),
                pb_utils.Tensor('length', length_np),
            ],
        )
        encoder_response = encoder_request.exec()
        if encoder_response.has_error():
            if _timer is not None:
                _timer.stop()
            raise pb_utils.TritonModelException(
                f'indic_asr_encoder BLS error: {encoder_response.error().message()}'
            )

        outputs = _pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(encoder_response, 'outputs'), 'outputs')
        encoded_lengths = _pb_tensor_to_numpy(
            pb_utils.get_output_tensor_by_name(encoder_response, 'encoded_lengths'),
            'encoded_lengths',
        )
        if _timer is not None:
            _timer.stop()
        return outputs, encoded_lengths

    def _ctc_decode(self, encoder_outputs, encoded_lengths, lang, compute_timestamps=None, _timer=None):
        # Stage 3: CTC decode (decoder ONNX run + log_softmax + argmax + collapse)
        if _timer is not None:
            _timer.start('decode')
        logprobs = self.models['ctc_decoder'].run(['logprobs'], {'encoder_output': encoder_outputs})[0]
        logprobs = torch.from_numpy(logprobs[:, :, self.language_masks[lang]]).log_softmax(dim=-1)

        # currently no batching
        indices = torch.argmax(logprobs[0], dim=-1)
        collapsed_indices = torch.unique_consecutive(indices, dim=-1)
        if _timer is not None:
            try:
                _timer.num_tokens = int((collapsed_indices != self.config.BLANK_ID).sum().item())
            except Exception:
                pass
            _timer.stop()

        # Stage 4: postprocessing (vocab lookup + text assembly + optional timestamps)
        if _timer is not None:
            _timer.start('postproc')
        hyp = ''.join([self.vocab[lang][x] for x in collapsed_indices if x != self.config.BLANK_ID]).replace('▁', ' ').strip()

        if compute_timestamps:
            result = (hyp, self.compute_timestamps(logprobs, encoded_lengths, lang, _type=compute_timestamps))
            if _timer is not None:
                _timer.stop()
            return result
        else:
            del logprobs, indices, collapsed_indices
            if _timer is not None:
                _timer.stop()
            return hyp

    def compute_timestamps(self, batch_logprobs, lens, lang, _type='w'):
        """Return a list of lists — one (token, t0, t1) tuple per contiguous token."""
        assert _type in ['w', 'c']
        results = []
        results_word = []
        for b in range(batch_logprobs.size(0)):
            T = lens[b].item()
            lp = batch_logprobs[b, :T]
            path = lp.argmax(dim=-1).cpu()
            step_sec = self.config.FRAME_DURATION_MS

            segments, cur_tok, start_f = [], None, 0
            segments_word = []
            for f, tok in enumerate(path):
                tok = tok.item()
                if tok == self.config.BLANK_ID:
                    if cur_tok is not None:
                        segments.append(
                            (self.vocab[lang][cur_tok], start_f * step_sec, f * step_sec)
                        )
                        cur_tok = None
                elif tok != cur_tok:
                    if cur_tok is not None:
                        segments.append(
                            (self.vocab[lang][cur_tok], start_f * step_sec, f * step_sec)
                        )
                    cur_tok, start_f = tok, f

            if cur_tok is not None:
                segments.append(
                    (self.vocab[lang][cur_tok], start_f * step_sec, T * step_sec)
                )

            if _type == 'w':
                word = ''
                start_t = None
                prev_t1 = 0

                for token, t0, t1 in segments:
                    if '▁' in token:
                        if word:
                            segments_word.append((word, start_t, prev_t1))
                        word = token.replace('▁', '')
                        start_t = t0
                    else:
                        word += token

                    prev_t1 = t1

                if word:
                    segments_word.append((word, start_t, prev_t1))

            results.append(segments)
            results_word.append(segments_word)
        return results if _type == 'c' else results_word

    def _rnnt_decode(self, encoder_outputs, encoded_lengths, lang, _timer=None):
        # Stage 3: RNNT decode (joint_enc projection + greedy loop over rnnt_decoder/joint_pred/joint_pre_net/joint_post_net)
        if _timer is not None:
            _timer.start('decode')
        joint_enc = self.models['joint_enc'].run(['output'], {'input': encoder_outputs.transpose(0, 2, 1)})[0]
        joint_enc = torch.from_numpy(joint_enc)
        hyp = [self.config.SOS]
        prev_dec_state = (
            np.zeros((self.config.PRED_RNN_LAYERS, 1, self.config.PRED_RNN_HIDDEN_DIM), dtype=np.float32),
            np.zeros((self.config.PRED_RNN_LAYERS, 1, self.config.PRED_RNN_HIDDEN_DIM), dtype=np.float32),
        )

        for t in range(joint_enc.size(1)):
            f = joint_enc[:, t, :].unsqueeze(1)

            not_blank = True
            symbols_added = 0

            while not_blank and ((self.config.RNNT_MAX_SYMBOLS is None) or (symbols_added < self.config.RNNT_MAX_SYMBOLS)):
                g, _, dec_state_0, dec_state_1 = self.models['rnnt_decoder'].run(
                    ['outputs', 'prednet_lengths', 'states', '162'],
                    {'targets': np.array([[hyp[-1]]], dtype=np.int32),
                     'target_length': np.array([1], dtype=np.int32),
                     'states.1': prev_dec_state[0],
                     'onnx::Slice_3': prev_dec_state[1]})

                g = self.models['joint_pred'].run(['output'], {'input': g.transpose(0, 2, 1)})[0]

                joint_out = f + g
                joint_out = self.models['joint_pre_net'].run(['output'], {'input': joint_out.numpy()})[0]

                logits = self.models[f'joint_post_net_{lang}'].run(['output'], {'input': joint_out})[0]
                log_probs = torch.from_numpy(logits).log_softmax(dim=-1)
                pred_token = log_probs.argmax(dim=-1).item()

                if pred_token == self.config.BLANK_ID:
                    not_blank = False
                else:
                    hyp.append(pred_token)
                    prev_dec_state = (dec_state_0, dec_state_1)

                symbols_added += 1

        if _timer is not None:
            try:
                _timer.num_tokens = max(0, len(hyp) - 1)  # exclude SOS
            except Exception:
                pass
            _timer.stop()

        # Stage 4: postprocessing (vocab lookup + text assembly)
        if _timer is not None:
            _timer.start('postproc')
        pred_text = ''.join([self.vocab[lang][x] for x in hyp if x != self.config.SOS]).replace('▁', ' ').strip()
        if _timer is not None:
            _timer.stop()
        return pred_text

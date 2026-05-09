import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.blank_idx = 0
        self.max_symbols_per_step = 30
        self.pred_layers, self.pred_hidden = 1, 640
        # Per-language vocab masks load here from a JSON shipped in 1/

    def execute(self, requests):
        responses = []
        for req in requests:
            audio = pb_utils.get_input_tensor_by_name(req, "AUDIO_SIGNAL").as_numpy()
            length = pb_utils.get_input_tensor_by_name(req, "LENGTH").as_numpy()
            lang = pb_utils.get_input_tensor_by_name(req, "LANGUAGE").as_numpy()[0].decode()

            # 1. Preproc — BLS call, tensors stay on the same process
            mel, mel_len = self._bls(
                "indic_asr_preproc",
                {"INPUT__0": audio, "INPUT__1": length},
                ["OUTPUT__0", "OUTPUT__1"],
            )

            # 2. Encoder — single TRT call, the ONE expensive op
            enc_out, enc_len = self._bls(
                "indic_asr_encoder",
                {"audio_signal": mel, "length": mel_len},
                ["outputs", "encoded_lengths"],
            )
            # enc_out: [B, 1024, T]

            B = enc_out.shape[0]
            tokens_per_b = [self._greedy_decode(enc_out[b:b+1], int(enc_len[b]), lang)
                            for b in range(B)]
            responses.append(self._pack(tokens_per_b))
        return responses

    def _greedy_decode(self, enc_b, T, lang):
        # enc_b: [1, 1024, T]
        tokens = []
        state_h = np.zeros((self.pred_layers, 1, self.pred_hidden), np.float32)
        state_c = np.zeros((self.pred_layers, 1, self.pred_hidden), np.float32)
        last = np.array([[self.blank_idx]], np.int64)
        pred_out, state_h, state_c = self._step_predictor(last, state_h, state_c)
        mask = self._lang_mask(lang)  # [V] additive -inf mask

        for t in range(T):
            enc_t = enc_b[:, :, t:t+1].transpose(0, 2, 1)  # [1, 1, 1024]
            for _ in range(self.max_symbols_per_step):
                logits = self._step_joint(enc_t, pred_out)[0, 0, 0]  # [V]
                tok = int(np.argmax(logits + mask))
                if tok == self.blank_idx:
                    break
                tokens.append(tok)
                last = np.array([[tok]], np.int64)
                pred_out, state_h, state_c = self._step_predictor(last, state_h, state_c)
        return tokens
    # _bls, _step_predictor, _step_joint, _lang_mask, _pack omitted for brevity

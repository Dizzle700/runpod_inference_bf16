#!/usr/bin/env python3
"""Launch vLLM with compatibility support for Audio Flamingo Next.

vLLM 0.20.x expects MusicFlamingo's Hugging Face processor to return
``rote_timestamps``. The public Audio Flamingo Next processor does not return
that field, which makes vLLM fail during multimodal profiling before the server
can start. Newer vLLM code derives the timestamps from the audio chunks. This
launcher applies that same fallback while retaining the pinned vLLM version
required by the RunPod image.
"""

from __future__ import annotations

import runpy


_PROCESSOR_ERROR = "MusicFlamingoProcessor output must include `rote_timestamps`."
_MODEL_ERROR = "MusicFlamingo audio feature inputs must include `rote_timestamps`."


def _patch_musicflamingo() -> None:
    import torch
    from vllm.model_executor.models import musicflamingo

    processor_cls = musicflamingo.MusicFlamingoMultiModalProcessor
    original_call_hf_processor = processor_cls._call_hf_processor

    def call_hf_processor_with_fallback(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        try:
            return original_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs)
        except KeyError as exc:
            if str(exc) != repr(_PROCESSOR_ERROR):
                raise

        # Repeat the upstream processing up to the old version's strict
        # `rote_timestamps` check. The model fallback below supplies the field.
        outputs = super(processor_cls, self)._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        audio_data = mm_data.get("audio")
        if audio_data is None:
            return outputs
        audio_list = audio_data if isinstance(audio_data, list) else [audio_data]
        if not audio_list:
            return outputs

        processor = self.info.get_hf_processor(**mm_kwargs)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        window_size = int(sampling_rate * feature_extractor.chunk_length)
        max_windows = int(processor.max_audio_len // feature_extractor.chunk_length)
        chunk_counts = []
        for audio in audio_list:
            n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]
            n_windows = max(1, (n_samples + window_size - 1) // window_size)
            chunk_counts.append(min(n_windows, max_windows))
        outputs["chunk_counts"] = torch.tensor(chunk_counts, dtype=torch.long)
        return outputs

    processor_cls._call_hf_processor = call_hf_processor_with_fallback

    model_cls = musicflamingo.MusicFlamingoForConditionalGeneration
    original_process_audio_input = model_cls._process_audio_input

    def process_audio_input_with_fallback(self, audio_input):
        try:
            return original_process_audio_input(self, audio_input)
        except ValueError as exc:
            if str(exc) != _MODEL_ERROR or audio_input["type"] == "audio_embeds":
                raise

        (
            input_features,
            feature_attention_mask,
            chunk_counts,
        ) = self._normalize_audio_feature_inputs(audio_input)
        hidden_states = self._encode_audio_features(input_features, feature_attention_mask)
        audio_frame_step = self.config.audio_frame_step * 4
        frame_offsets = (
            torch.arange(hidden_states.shape[-2], device=hidden_states.device, dtype=torch.float32)
            * audio_frame_step
        )
        if chunk_counts:
            window_indices = torch.cat(
                [torch.arange(count, device=hidden_states.device, dtype=torch.float32) for count in chunk_counts]
            )
            rote_timestamps = (
                window_indices.unsqueeze(1) * hidden_states.shape[-2] * audio_frame_step + frame_offsets
            )
        else:
            rote_timestamps = frame_offsets.new_empty((0, hidden_states.shape[-2]))

        cos, sin = self.pos_emb(rote_timestamps, seq_len=hidden_states.shape[-2])
        hidden_states = musicflamingo.apply_rotary_time_emb(hidden_states, cos, sin)
        audio_features = self.multi_modal_projector(hidden_states)
        return self._group_audio_embeddings(audio_features, feature_attention_mask, chunk_counts)

    model_cls._process_audio_input = process_audio_input_with_fallback


def main() -> None:
    _patch_musicflamingo()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

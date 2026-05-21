# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for _fix_bpe_boundary_orphans — the fix for PyTorch issue #184431.

The bug: when <prompt_embeds> is registered as a special token, the HF
tokenizer splits text at its boundary before applying BPE merges. This
prevents cross-boundary merges (e.g. ".\\n" → single token) and leaves
orphan tokens that shift positions by +1, causing RoPE divergence.

The fix: compare the token count of the window around the placeholder
against a reference tokenization with source_text, and remove orphans.
"""

from __future__ import annotations

import io
from typing import Final
from unittest import mock

import pybase64 as base64
import pytest
import torch
from transformers import AutoTokenizer

from vllm.entrypoints.chat_utils import (
    PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
    parse_chat_messages,
)
from vllm.renderers.hf import (
    _ensure_prompt_embeds_placeholder_token,
    _fix_bpe_boundary_orphans,
)

TOKENIZER_IDS: Final[list[str]] = [
    "gpt2",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

_HIDDEN_SIZE: Final[int] = 8
_DTYPE: Final[torch.dtype] = torch.float32


@pytest.fixture(params=TOKENIZER_IDS, ids=TOKENIZER_IDS)
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


def _make_model_config():
    mc = mock.MagicMock()
    mc.enable_prompt_embeds = True
    mc.multimodal_config = None
    mc.allowed_local_media_path = None
    mc.allowed_media_domains = None
    mc.is_multimodal_model = False
    mc.get_hidden_size.return_value = _HIDDEN_SIZE
    mc.dtype = _DTYPE
    return mc


def _encode_tensor(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestFixBpeBoundaryOrphans:
    """Direct unit tests for _fix_bpe_boundary_orphans."""

    @pytest.mark.parametrize(
        "source_text",
        [
            "Describe this audio.",
            "Listen carefully!",
            "Tell me about this.",
            "What?",
        ],
        ids=["period", "exclamation", "period2", "question"],
    )
    def test_merge_case_removes_orphan(self, tokenizer, source_text):
        """Source texts ending in punctuation that merges with \\n should
        have the orphan \\n removed after the placeholder."""
        ph_token_id = _ensure_prompt_embeds_placeholder_token(tokenizer)
        PE = PROMPT_EMBEDS_PLACEHOLDER_TOKEN

        embeds_text = f"user\n{PE}\nAudio 1: test"
        embeds_ids = tokenizer.encode(embeds_text, add_special_tokens=False)

        raw_text = f"user\n{source_text}\nAudio 1: test"
        raw_ids = tokenizer.encode(raw_text, add_special_tokens=False)

        source_ids = tokenizer.encode(source_text, add_special_tokens=False)
        tensor_length = len(source_ids)

        fixed = _fix_bpe_boundary_orphans(
            embeds_ids,
            ph_token_id,
            [tensor_length],
            [source_text],
            tokenizer,
        )

        expanded_count = len(fixed) - 1 + tensor_length
        assert expanded_count == len(raw_ids), (
            f"source_text={source_text!r}: "
            f"expanded={expanded_count} != raw={len(raw_ids)}"
        )

    @pytest.mark.parametrize(
        "source_text",
        ["Hello", "a", "test123", "ok"],
        ids=["Hello", "single-a", "digits", "ok"],
    )
    def test_no_merge_case_preserves_tokens(self, tokenizer, source_text):
        """Source texts NOT ending in merge-eligible chars should leave
        the token stream unchanged."""
        ph_token_id = _ensure_prompt_embeds_placeholder_token(tokenizer)
        PE = PROMPT_EMBEDS_PLACEHOLDER_TOKEN

        embeds_text = f"user\n{PE}\nAudio 1: test"
        embeds_ids = tokenizer.encode(embeds_text, add_special_tokens=False)

        raw_text = f"user\n{source_text}\nAudio 1: test"
        raw_ids = tokenizer.encode(raw_text, add_special_tokens=False)

        source_ids = tokenizer.encode(source_text, add_special_tokens=False)
        tensor_length = len(source_ids)

        fixed = _fix_bpe_boundary_orphans(
            embeds_ids,
            ph_token_id,
            [tensor_length],
            [source_text],
            tokenizer,
        )

        expanded_count = len(fixed) - 1 + tensor_length
        assert expanded_count == len(raw_ids), (
            f"source_text={source_text!r}: "
            f"expanded={expanded_count} != raw={len(raw_ids)}"
        )

    def test_multiple_placeholders(self, tokenizer):
        """Multiple placeholders should each be independently corrected."""
        ph_token_id = _ensure_prompt_embeds_placeholder_token(tokenizer)
        PE = PROMPT_EMBEDS_PLACEHOLDER_TOKEN

        embeds_text = f"user\n{PE}\nAudio 1:\n{PE}\nDone"
        embeds_ids = tokenizer.encode(embeds_text, add_special_tokens=False)

        source_text_1 = "Describe this audio."
        source_text_2 = "Listen here."
        raw_text = f"user\n{source_text_1}\nAudio 1:\n{source_text_2}\nDone"
        raw_ids = tokenizer.encode(raw_text, add_special_tokens=False)

        source_ids_1 = tokenizer.encode(source_text_1, add_special_tokens=False)
        source_ids_2 = tokenizer.encode(source_text_2, add_special_tokens=False)

        fixed = _fix_bpe_boundary_orphans(
            embeds_ids,
            ph_token_id,
            [len(source_ids_1), len(source_ids_2)],
            [source_text_1, source_text_2],
            tokenizer,
        )

        expanded_count = len(fixed) - 2 + len(source_ids_1) + len(source_ids_2)
        assert expanded_count == len(raw_ids)

    def test_source_text_none_skipped(self, tokenizer):
        """When source_text is None, the placeholder should not be modified."""
        ph_token_id = _ensure_prompt_embeds_placeholder_token(tokenizer)
        PE = PROMPT_EMBEDS_PLACEHOLDER_TOKEN

        embeds_text = f"user\n{PE}\nAudio 1: test"
        embeds_ids = tokenizer.encode(embeds_text, add_special_tokens=False)

        fixed = _fix_bpe_boundary_orphans(
            embeds_ids,
            ph_token_id,
            [4],
            [None],
            tokenizer,
        )

        assert fixed == embeds_ids

    def test_empty_source_text(self, tokenizer):
        """Empty source_text should not crash."""
        ph_token_id = _ensure_prompt_embeds_placeholder_token(tokenizer)
        PE = PROMPT_EMBEDS_PLACEHOLDER_TOKEN

        embeds_text = f"user\n{PE}\nAudio"
        embeds_ids = tokenizer.encode(embeds_text, add_special_tokens=False)

        raw_text = "user\n\nAudio"
        raw_ids = tokenizer.encode(raw_text, add_special_tokens=False)

        fixed = _fix_bpe_boundary_orphans(
            embeds_ids,
            ph_token_id,
            [0],
            [""],
            tokenizer,
        )

        expanded_count = len(fixed) - 1 + 0
        assert expanded_count == len(raw_ids)

    def test_audio_first_ordering(self, tokenizer):
        """audio_first=True: placeholder after audio content."""
        ph_token_id = _ensure_prompt_embeds_placeholder_token(tokenizer)
        PE = PROMPT_EMBEDS_PLACEHOLDER_TOKEN

        embeds_text = f"user\nAudio 1: test\n{PE}\nend"
        embeds_ids = tokenizer.encode(embeds_text, add_special_tokens=False)

        source_text = "Describe this audio."
        raw_text = f"user\nAudio 1: test\n{source_text}\nend"
        raw_ids = tokenizer.encode(raw_text, add_special_tokens=False)

        source_ids = tokenizer.encode(source_text, add_special_tokens=False)
        tensor_length = len(source_ids)

        fixed = _fix_bpe_boundary_orphans(
            embeds_ids,
            ph_token_id,
            [tensor_length],
            [source_text],
            tokenizer,
        )

        expanded_count = len(fixed) - 1 + tensor_length
        assert expanded_count == len(raw_ids)

    def test_mismatched_placeholder_count_returns_unchanged(self, tokenizer):
        """If placeholder count != tensor_lengths count, return unchanged."""
        ph_token_id = _ensure_prompt_embeds_placeholder_token(tokenizer)
        PE = PROMPT_EMBEDS_PLACEHOLDER_TOKEN

        embeds_text = f"user\n{PE}\ntest"
        embeds_ids = tokenizer.encode(embeds_text, add_special_tokens=False)

        # Provide 2 tensor_lengths but there's only 1 placeholder
        fixed = _fix_bpe_boundary_orphans(
            embeds_ids,
            ph_token_id,
            [4, 3],
            ["foo.", "bar."],
            tokenizer,
        )

        assert fixed == embeds_ids


class TestSourceTextParsing:
    """Test that source_text flows through the parsing pipeline."""

    def test_source_text_in_openai_format(self):
        mc = _make_model_config()
        tensor = torch.randn(4, _HIDDEN_SIZE, dtype=_DTYPE)
        b64 = _encode_tensor(tensor)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "prompt_embeds",
                        "data": b64,
                        "source_text": "Describe this audio.",
                    },
                    {"type": "text", "text": "What do you hear?"},
                ],
            }
        ]

        _, mm_data, _ = parse_chat_messages(messages, mc, content_format="openai")
        assert "prompt_embeds_source_texts" in mm_data
        assert mm_data["prompt_embeds_source_texts"] == ["Describe this audio."]

    def test_no_source_text_backward_compat(self):
        mc = _make_model_config()
        tensor = torch.randn(4, _HIDDEN_SIZE, dtype=_DTYPE)
        b64 = _encode_tensor(tensor)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": b64},
                    {"type": "text", "text": "What do you hear?"},
                ],
            }
        ]

        _, mm_data, _ = parse_chat_messages(messages, mc, content_format="openai")
        assert "prompt_embeds_source_texts" not in mm_data

    def test_mixed_source_text_presence(self):
        mc = _make_model_config()
        t1 = torch.randn(4, _HIDDEN_SIZE, dtype=_DTYPE)
        t2 = torch.randn(3, _HIDDEN_SIZE, dtype=_DTYPE)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "prompt_embeds",
                        "data": _encode_tensor(t1),
                        "source_text": "First.",
                    },
                    {"type": "text", "text": "middle"},
                    {"type": "prompt_embeds", "data": _encode_tensor(t2)},
                ],
            }
        ]

        _, mm_data, _ = parse_chat_messages(messages, mc, content_format="openai")
        assert mm_data["prompt_embeds_source_texts"] == ["First.", None]

    def test_source_text_in_string_format(self):
        mc = _make_model_config()
        tensor = torch.randn(4, _HIDDEN_SIZE, dtype=_DTYPE)
        b64 = _encode_tensor(tensor)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "prompt_embeds",
                        "data": b64,
                        "source_text": "Hello world.",
                    },
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]

        _, mm_data, _ = parse_chat_messages(messages, mc, content_format="string")
        assert mm_data["prompt_embeds_source_texts"] == ["Hello world."]

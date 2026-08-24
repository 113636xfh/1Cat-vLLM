# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model serving defaults: fill-in, per-request override, bounds, and
fail-closed processor validation (DoD: a client gets the model's recipe by
default without knowing any engine knobs)."""

import pytest

from vllm.entrypoints.openai.serving_defaults import (
    ModelServingDefaults,
    get_model_serving_defaults,
    merge_extra_args,
    register_model_serving_defaults,
    validate_required_logits_processors,
)

pytestmark = pytest.mark.cpu_test


def _defaults() -> ModelServingDefaults:
    return ModelServingDefaults(
        sampling_defaults={"temperature": 0.0, "skip_special_tokens": False},
        extra_args_defaults={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        },
        extra_args_bounds={"ngram_size": (1, 512), "window_size": (1, 8192)},
        required_logits_processors=("pkg.mod.NGramPerReqLogitsProcessor",),
    )


def test_deepseek_ocr_registered_in_core():
    defaults = get_model_serving_defaults(["DeepseekOCRForCausalLM"])
    assert defaults is not None
    assert defaults.extra_args_defaults["ngram_size"] == 30
    assert defaults.extra_args_defaults["window_size"] == 90
    assert defaults.sampling_defaults["skip_special_tokens"] is False
    assert defaults.sampling_defaults["temperature"] == 0.0


def test_unknown_architecture_has_no_defaults():
    assert get_model_serving_defaults(["LlamaForCausalLM"]) is None
    assert get_model_serving_defaults(None) is None


def test_merge_fills_missing_keys():
    merged = merge_extra_args(_defaults(), None)
    assert merged["ngram_size"] == 30
    assert merged["window_size"] == 90
    assert merged["whitelist_token_ids"] == [128821, 128822]


def test_merge_request_overrides_within_bounds():
    merged = merge_extra_args(_defaults(), {"ngram_size": 20, "other": "x"})
    assert merged["ngram_size"] == 20  # client override wins
    assert merged["window_size"] == 90  # default preserved
    assert merged["other"] == "x"  # unbounded keys pass through


@pytest.mark.parametrize("bad", [0, -5, 100000, "30", 3.5, True])
def test_merge_rejects_out_of_bounds_or_non_integer(bad):
    with pytest.raises(ValueError):
        merge_extra_args(_defaults(), {"ngram_size": bad})


def test_required_processor_validation_passes_by_qualname_or_class_name():
    validate_required_logits_processors(
        _defaults(), ["pkg.mod.NGramPerReqLogitsProcessor"], "m"
    )

    class NGramPerReqLogitsProcessor:  # matched by bare class name
        pass

    validate_required_logits_processors(_defaults(), [NGramPerReqLogitsProcessor], "m")


def test_required_processor_validation_fails_closed_when_missing():
    with pytest.raises(ValueError, match="logits processor"):
        validate_required_logits_processors(_defaults(), None, "m")
    with pytest.raises(ValueError, match="logits processor"):
        validate_required_logits_processors(_defaults(), ["OtherProcessor"], "m")


def test_register_and_resolve_roundtrip():
    register_model_serving_defaults(
        "TestArchForCausalLM", ModelServingDefaults(sampling_defaults={"top_k": 1})
    )
    got = get_model_serving_defaults(["OtherArch", "TestArchForCausalLM"])
    assert got is not None and got.sampling_defaults == {"top_k": 1}

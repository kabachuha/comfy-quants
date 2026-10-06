"""Kandinsky 6 model adapter.

Authored from ComfyUI ``comfy/ldm/kandinsky6/model.py``, the release
``core_contract`` / ``detection`` and the Pro ``transformer_pro.json`` module
config, and validated against the joint video+audio tensor architecture.
Kandinsky 6 fuses one video and one audio transformer stream per block
(``visual_blocks``) around shared text-encoder streams.
"""

from __future__ import annotations

import json
from pathlib import Path

from comfy_quants.comfy.stock_dit_contract import stock_dit_artifact_contract_metadata
from comfy_quants.core.policy import QuantPolicy
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.stock_dit_contract import (
    BlockGroup,
    StockDitContract,
    build_stock_dit_graph,
    kept_component,
    linear,
    summarize_stock_dit_graph,
)

CONTRACT_SCHEMA_VERSION = "kandinsky6_static_contract.v1"

# Dimensions shared by every released variant.
_IN_VISUAL_DIM = 16   # in_visual_dim
_PATCH_VOLUME = 4     # prod(patch_size=(1, 2, 2))

# Variant tables keyed by model_dim, mirroring ComfyUI kandinsky6 detection
# (``_RELEASE_DIT_CONFIGS``). Every field matches transformer_pro / transformer_lite.
_VARIANTS: dict[int, dict[str, int]] = {
    4096: {"model_dim": 4096, "model_dim_a": 2048, "ff_dim": 16384, "ff_dim_a": 7168,
           "time_dim": 1024, "time_dim_a": 1024, "num_text_blocks": 4, "num_visual_blocks": 60},
    1792: {"model_dim": 1792, "model_dim_a": 896, "ff_dim": 7168, "ff_dim_a": 3584,
           "time_dim": 512, "time_dim_a": 512, "num_text_blocks": 2, "num_visual_blocks": 32},
}
_VARIANT_NAMES = {4096: "pro", 1792: "lite"}
_DEFAULT_MODEL_DIM = 4096  # Pro


def _variant(model_dim: int) -> dict[str, int]:
    return _VARIANTS.get(int(model_dim), _VARIANTS[_DEFAULT_MODEL_DIM])


def _dims(model_dim: int) -> dict[str, int]:
    v = _variant(model_dim)
    h, ha = v["model_dim"], v["model_dim_a"]
    return {
        "H": h, "HA": ha, "FF": v["ff_dim"], "FFA": v["ff_dim_a"],
        "TD": v["time_dim"], "TDA": v["time_dim_a"],
        "VIS_EMBED": (2 * _IN_VISUAL_DIM + 1) * _PATCH_VOLUME,
        "VIS_MOD": 9 * h,
        "VIS_MOD_A": 9 * ha,
        "VA_MOD": 2 * h + ha,
        "AV_MOD": 2 * ha + h,
    }


def _block_counts(model_dim: int) -> tuple[int, int]:
    v = _variant(model_dim)
    return v["num_text_blocks"], v["num_visual_blocks"]


def _model_dim_from_source(source: ModelSource) -> int | None:
    """Read model_dim from a local safetensors header, else None (Pro default).

    The visual input projection's out_features equals model_dim. Only the
    8-byte little-endian header length + JSON header are read (no tensor data,
    no torch), so a single-file checkpoint is probed without loading weights.
    """
    path = Path(source.model_id)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len))
    except (OSError, ValueError):
        return None
    suffix = "visual_embeddings.in_layer.weight"
    for name, info in header.items():
        if name == suffix or name.endswith("." + suffix):
            shape = (info or {}).get("shape") or []
            if shape:
                return int(shape[0])
    return None


def _video_text_block_modules() -> tuple:
    p = "video_text_transformer_blocks.{block}"
    return (
        linear(f"{p}.feed_forward.in_layer", "FF", "H"),
        linear(f"{p}.feed_forward.out_layer", "H", "FF"),
        linear(f"{p}.self_attention.to_query", "H", "H"),
        linear(f"{p}.self_attention.to_key", "H", "H"),
        linear(f"{p}.self_attention.to_value", "H", "H"),
        linear(f"{p}.self_attention.out_layer", "H", "H"),
    )


def _audio_text_block_modules() -> tuple:
    p = "audio_text_transformer_blocks.{block}"
    return (
        linear(f"{p}.feed_forward.in_layer", "FFA", "HA"),
        linear(f"{p}.feed_forward.out_layer", "HA", "FFA"),
        linear(f"{p}.self_attention.to_query", "HA", "HA"),
        linear(f"{p}.self_attention.to_key", "HA", "HA"),
        linear(f"{p}.self_attention.to_value", "HA", "HA"),
        linear(f"{p}.self_attention.out_layer", "HA", "HA"),
    )


def _visual_block_modules() -> tuple:
    p = "visual_blocks.{block}"
    return (
        # video decoder
        linear(f"{p}.videoT.self_attention.to_query", "H", "H"),
        linear(f"{p}.videoT.self_attention.to_key", "H", "H"),
        linear(f"{p}.videoT.self_attention.to_value", "H", "H"),
        linear(f"{p}.videoT.self_attention.out_layer", "H", "H"),
        linear(f"{p}.videoT.cross_attention.to_query", "H", "H"),
        linear(f"{p}.videoT.cross_attention.to_key", "H", "H"),
        linear(f"{p}.videoT.cross_attention.to_value", "H", "H"),
        linear(f"{p}.videoT.cross_attention.out_layer", "H", "H"),
        linear(f"{p}.videoT.feed_forward.in_layer", "FF", "H"),
        linear(f"{p}.videoT.feed_forward.out_layer", "H", "FF"),
        linear(f"{p}.videoT.visual_modulation.out_layer", "VIS_MOD", "TD"),
        # audio decoder
        linear(f"{p}.audioT.self_attention.to_query", "HA", "HA"),
        linear(f"{p}.audioT.self_attention.to_key", "HA", "HA"),
        linear(f"{p}.audioT.self_attention.to_value", "HA", "HA"),
        linear(f"{p}.audioT.self_attention.out_layer", "HA", "HA"),
        linear(f"{p}.audioT.cross_attention.to_query", "HA", "HA"),
        linear(f"{p}.audioT.cross_attention.to_key", "HA", "HA"),
        linear(f"{p}.audioT.cross_attention.to_value", "HA", "HA"),
        linear(f"{p}.audioT.cross_attention.out_layer", "HA", "HA"),
        linear(f"{p}.audioT.feed_forward.in_layer", "FFA", "HA"),
        linear(f"{p}.audioT.feed_forward.out_layer", "HA", "FFA"),
        linear(f"{p}.audioT.visual_modulation.out_layer", "VIS_MOD_A", "TDA"),
        # video -> audio cross attention (query from video, key/value from audio)
        linear(f"{p}.va_cross_attention.to_query", "H", "H"),
        linear(f"{p}.va_cross_attention.to_key", "H", "HA"),
        linear(f"{p}.va_cross_attention.to_value", "H", "HA"),
        linear(f"{p}.va_cross_attention.out_layer", "H", "H"),
        # audio -> video cross attention (query from audio, key/value from video)
        linear(f"{p}.av_cross_attention.to_query", "HA", "HA"),
        linear(f"{p}.av_cross_attention.to_key", "HA", "H"),
        linear(f"{p}.av_cross_attention.to_value", "HA", "H"),
        linear(f"{p}.av_cross_attention.out_layer", "HA", "HA"),
        # cross-stream AdaLN (cross_gates layout)
        linear(f"{p}.va_modulation.out_layer", "VA_MOD", "TD"),
        linear(f"{p}.av_modulation.out_layer", "AV_MOD", "TDA"),
    )


def _extra_components() -> tuple:
    return (
        kept_component("visual_embeddings", "Linear", "transformer", "video input projection kept high precision"),
        kept_component("visual_token_type_embeddings", "Embedding", "transformer", "I2VA token-type embeddings kept high precision"),
        kept_component("video_time_embeddings", "MLPEmbedder", "transformer", "video timestep embedding kept high precision"),
        kept_component("video_text_embeddings", "Linear", "transformer", "video text input projection kept high precision"),
        kept_component("video_pooled_text_embeddings", "Linear", "transformer", "video pooled text projection kept high precision"),
        kept_component("out_layer", "LastLayer", "transformer", "video final layer kept high precision"),
        kept_component("audio_embeddings", "Linear", "transformer", "audio input projection kept high precision"),
        kept_component("audio_time_embeddings", "MLPEmbedder", "transformer", "audio timestep embedding kept high precision"),
        kept_component("audio_text_embeddings", "Linear", "transformer", "audio text input projection kept high precision"),
        kept_component("audio_pooled_text_embeddings", "Linear", "transformer", "audio pooled text projection kept high precision"),
        kept_component("audio_outLayer", "Linear", "transformer", "audio final layer kept high precision"),
    )


def build_kandinsky6_static_contract(model_dim: int = _DEFAULT_MODEL_DIM) -> StockDitContract:
    v = _variant(model_dim)
    return StockDitContract(
        family="kandinsky6",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="int8_tensorwise",
        dims=_dims(model_dim),
        block_groups=(
            BlockGroup(prefix="video_text_transformer_blocks", count=v["num_text_blocks"], modules=_video_text_block_modules()),
            BlockGroup(prefix="audio_text_transformer_blocks", count=v["num_text_blocks"], modules=_audio_text_block_modules()),
            BlockGroup(prefix="visual_blocks", count=v["num_visual_blocks"], modules=_visual_block_modules()),
        ),
        extra_components=_extra_components(),
        metadata={
            "export_name": "Kandinsky 6",
            "architecture": "fused_av_transformer",
            "variant": _VARIANT_NAMES.get(int(model_dim), "pro"),
            "model_dim": v["model_dim"],
            "model_dim_a": v["model_dim_a"],
            "num_text_blocks": v["num_text_blocks"],
            "num_visual_blocks": v["num_visual_blocks"],
            "ff_dim": v["ff_dim"],
            "ff_dim_a": v["ff_dim_a"],
            "is_multimodal": True,
        },
    )


class Kandinsky6Adapter:
    """Adapter for the Kandinsky 6 joint video+audio DiT."""

    family = "kandinsky6"
    supported_model_ids = ["kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"]

    def inspect(self, source: ModelSource):
        model_dim = _model_dim_from_source(source) or _DEFAULT_MODEL_DIM
        contract = build_kandinsky6_static_contract(model_dim)
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("kandinsky6"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "int8_tensorwise", *, mixed: bool = False) -> QuantPolicy:
        """Return the layer-selection policy for Kandinsky 6.

        With mixed=True the cross-stream attention is protected to preserve
        video/audio coherence.
        """
        return QuantPolicy(
            name="kandinsky6_mixed" if mixed else "kandinsky6_default",
            algorithm="int8_tensorwise",
            target_dtype=target_dtype,
            include=[
                "video_text_transformer_blocks.*",
                "audio_text_transformer_blocks.*",
                "visual_blocks.*",
            ],
            exclude=[
                "visual_blocks.*.va_cross_attention.*",
                "visual_blocks.*.av_cross_attention.*",
            ] if mixed else [],
            keep_components=["visual_embeddings", "audio_embeddings"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(Kandinsky6Adapter())

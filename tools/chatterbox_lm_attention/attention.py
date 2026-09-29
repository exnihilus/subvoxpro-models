"""Standard-operator attention with a fixed-capacity KV cache, replacing com.microsoft.GroupQueryAttention.

DirectML re-prepares a whole graph whenever an input shape changes, so the KV cache must keep its shape from
one step to the next. Instead of appending, each layer writes the new keys/values into a cache of fixed capacity:

- `attention_mask` is [batch, capacity] and positional: 1 marks a slot holding a real token, 0 hides it.
- `cache_write_index` ([1], int64) is the slot where the first new token goes; the new tokens fill the next slots.
- The capacity is the mask width. When it exceeds the incoming cache length, the cache is zero-padded first,
  so the runtime grows the cache in large steps and every other step keeps the same shapes.

The rotary position of a token is its rank among valid tokens (cumsum(mask) - 1), which matches
GroupQueryAttention's positions for right-padded prompts whose generated tokens follow the padding.
"""
from dataclasses import dataclass

from onnx import TensorProto

MASKED_SCORE = -1e9
WRITE_INDEX_INPUT = "cache_write_index"


@dataclass
class AttentionShape:
    num_heads: int
    head_size: int
    scale: float

    @property
    def hidden_size(self):
        return self.num_heads * self.head_size

    @property
    def rotary_half(self):
        return self.head_size // 2


@dataclass
class SharedContext:
    bias: str
    cos: str
    sin: str
    cache_padding: str
    slot_placement: str
    slot_keep: str


def build_shared_context(builder, attention_mask, inputs_embeds, first_past_key, cos_cache, sin_cache):
    capacity = builder.op("Shape", [attention_mask], "capacity", start=1, end=2)
    new_length = builder.op("Shape", [inputs_embeds], "new_length", start=1, end=2)
    new_slots = _new_slots(builder, new_length)
    placement, keep = _slot_selection(builder, capacity, new_slots)
    return SharedContext(
        bias=_attention_bias(builder, attention_mask, capacity, new_slots),
        cos=_rotary_table(builder, cos_cache, attention_mask, new_slots, "cos"),
        sin=_rotary_table(builder, sin_cache, attention_mask, new_slots, "sin"),
        cache_padding=_cache_padding(builder, first_past_key, capacity),
        slot_placement=placement,
        slot_keep=keep)


def _new_slots(builder, new_length):
    count = builder.op("Squeeze", [new_length, builder.int64([0])], "new_count")
    offsets = builder.op("Range", [builder.int64(0), count, builder.int64(1)], "new_offsets")
    return builder.op("Add", [offsets, WRITE_INDEX_INPUT], "new_slots")


def _attention_bias(builder, attention_mask, capacity, new_slots):
    mask = builder.op("Cast", [attention_mask], "mask_float", to=TensorProto.FLOAT)
    hidden = builder.op("Sub", [builder.float32(1.0), mask], "mask_hidden")
    key_bias = builder.op("Mul", [hidden, builder.float32(MASKED_SCORE)], "key_bias")
    key_bias = builder.op("Unsqueeze", [key_bias, builder.int64([1, 2])], "key_bias_4d")

    capacity_scalar = builder.op("Squeeze", [capacity, builder.int64([0])], "capacity_scalar")
    key_index = builder.op("Range", [builder.int64(0), capacity_scalar, builder.int64(1)], "key_index")
    key_row = builder.op("Unsqueeze", [key_index, builder.int64([0])], "key_row")
    query_column = builder.op("Unsqueeze", [new_slots, builder.int64([1])], "query_column")
    future = builder.op("Greater", [key_row, query_column], "future_key")
    causal = builder.op("Where", [future, builder.float32(MASKED_SCORE), builder.float32(0.0)], "causal_bias")
    causal = builder.op("Unsqueeze", [causal, builder.int64([0, 1])], "causal_bias_4d")
    return builder.op("Add", [key_bias, causal], "attention_bias")


def _rotary_table(builder, cache, attention_mask, new_slots, label):
    rank = builder.op("CumSum", [attention_mask, builder.int64(1)], f"{label}_rank")
    position = builder.op("Sub", [rank, builder.int64(1)], f"{label}_position")
    position = builder.op("Gather", [position, new_slots], f"{label}_new_position", axis=1)
    table = builder.op("Gather", [cache, position], f"{label}_gather", axis=0)
    return builder.op("Unsqueeze", [table, builder.int64([1])], f"{label}_heads")


def _slot_selection(builder, capacity, new_slots):
    """One-hot placement [1, 1, capacity, new] of the new tokens in the cache, and the [1, 1, capacity, 1] mask
    of the slots they leave untouched. Written with MatMul/Mul so DirectML runs it (it lacks ScatterElements)."""
    capacity_scalar = builder.op("Squeeze", [capacity, builder.int64([0])], "slot_capacity")
    slot_index = builder.op("Range", [builder.int64(0), capacity_scalar, builder.int64(1)], "slot_index")
    slot_column = builder.op("Unsqueeze", [slot_index, builder.int64([1])], "slot_column")
    new_row = builder.op("Unsqueeze", [new_slots, builder.int64([0])], "new_row")
    matches = builder.op("Equal", [slot_column, new_row], "slot_matches")
    placement = builder.op("Cast", [matches], "slot_placement_2d", to=TensorProto.FLOAT)
    written = builder.op("ReduceSum", [placement, builder.int64([1])], "slot_written", keepdims=1)
    keep = builder.op("Sub", [builder.float32(1.0), written], "slot_keep_2d")
    placement = builder.op("Unsqueeze", [placement, builder.int64([0, 1])], "slot_placement")
    keep = builder.op("Unsqueeze", [keep, builder.int64([0, 1])], "slot_keep")
    return placement, keep


def _cache_padding(builder, past_key, capacity):
    past_length = builder.op("Shape", [past_key], "past_length", start=2, end=3)
    growth = builder.op("Sub", [capacity, past_length], "cache_growth")
    return builder.op(
        "Concat", [builder.int64([0, 0, 0, 0, 0, 0]), growth, builder.int64([0])], "cache_padding", axis=0)


def build_layer_attention(builder, shape, context, qkv, past_key, past_value, output, present_key, present_value):
    query, key, value = builder.op(
        "Split", [qkv, builder.int64([shape.hidden_size] * 3)], "split_qkv",
        outputs=[builder.name(n) for n in ("query", "key", "value")], axis=2)
    query = _rotate(builder, shape, context, _to_heads(builder, shape, query))
    key = _rotate(builder, shape, context, _to_heads(builder, shape, key))
    value = _to_heads(builder, shape, value)

    _write_cache(builder, context, past_key, key, present_key)
    _write_cache(builder, context, past_value, value, present_value)

    key_transposed = builder.op("Transpose", [present_key], "key_transposed", perm=[0, 1, 3, 2])
    scores = builder.op("MatMul", [query, key_transposed], "scores")
    scores = builder.op("Mul", [scores, builder.float32(shape.scale)], "scaled_scores")
    scores = builder.op("Add", [scores, context.bias], "masked_scores")
    weights = builder.op("Softmax", [scores], "attention_weights", axis=-1)
    attended = builder.op("MatMul", [weights, present_value], "attended")
    attended = builder.op("Transpose", [attended], "attended_tokens", perm=[0, 2, 1, 3])
    builder.op("Reshape", [attended, builder.int64([0, 0, shape.hidden_size])], "merge_heads", outputs=[output])


def _write_cache(builder, context, past, new, present):
    grown = builder.op("Pad", [past, context.cache_padding], "grown_cache", mode="constant")
    kept = builder.op("Mul", [grown, context.slot_keep], "kept_cache")
    placed = builder.op("MatMul", [context.slot_placement, new], "placed_tokens")
    builder.op("Add", [kept, placed], "write_cache", outputs=[present])


def _to_heads(builder, shape, tensor):
    split = builder.op("Reshape", [tensor, builder.int64([0, 0, shape.num_heads, shape.head_size])], "split_heads")
    return builder.op("Transpose", [split], "heads_first", perm=[0, 2, 1, 3])


def _rotate(builder, shape, context, tensor):
    half = shape.rotary_half
    first = builder.op("Slice", [tensor, builder.int64([0]), builder.int64([half]), builder.int64([3])], "rotary_first")
    second = builder.op(
        "Slice", [tensor, builder.int64([half]), builder.int64([shape.head_size]), builder.int64([3])], "rotary_second")
    first_cos = builder.op("Mul", [first, context.cos], "first_cos")
    second_sin = builder.op("Mul", [second, context.sin], "second_sin")
    second_cos = builder.op("Mul", [second, context.cos], "second_cos")
    first_sin = builder.op("Mul", [first, context.sin], "first_sin")
    rotated_first = builder.op("Sub", [first_cos, second_sin], "rotated_first")
    rotated_second = builder.op("Add", [second_cos, first_sin], "rotated_second")
    return builder.op("Concat", [rotated_first, rotated_second], "rotated", axis=3)

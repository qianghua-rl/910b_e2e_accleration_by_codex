"""Non-aligned hybrid-cache routing contract for the HF Agent runner.

This prototype owns per-agent state at arbitrary token boundaries. It only
chooses reuse versus fresh fallback; the correctness path remains tokenwise.
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentHybridState:
    token_ids: tuple[int, ...]
    cache: Any
    conv_states: Any
    recurrent_states: Any
    kv_cache: Any
    cache_attention_mask: tuple[int, ...] | None = None

    @property
    def sequence_length(self):
        return len(self.token_ids)

    def append(self, token_ids, cache):
        """Return a new boundary state after an exact suffix update."""
        suffix = tuple(int(x) for x in token_ids)
        next_mask = None
        if self.cache_attention_mask is not None:
            next_mask = self.cache_attention_mask + (1,) * len(suffix)
        return save_state(self.token_ids + suffix, cache, next_mask)


class AgentHybridStateStore:
    """Own one exact cache boundary per agent without alignment or compaction."""

    def __init__(self):
        self._states: dict[str, AgentHybridState] = {}

    def get(self, agent_id: str) -> AgentHybridState | None:
        return self._states.get(str(agent_id))

    def put(self, agent_id: str, state: AgentHybridState) -> AgentHybridState:
        if not isinstance(state, AgentHybridState):
            raise TypeError("state must be an AgentHybridState")
        self._states[str(agent_id)] = state
        return state

    def discard(self, agent_id: str) -> None:
        self._states.pop(str(agent_id), None)

    def __len__(self) -> int:
        return len(self._states)

    def route(self, agent_id: str, next_prompt, last_consumed: int):
        return choose_route(self.get(agent_id), next_prompt, last_consumed)

    def update(self, agent_id: str, next_prompt, last_consumed: int, suffix,
               updated_cache, fresh_cache=None):
        decision, state = append_or_fresh(
            self.get(agent_id), next_prompt, last_consumed, suffix,
            updated_cache, fresh_cache=fresh_cache)
        self.put(agent_id, state)
        return decision, state

    def import_batch_cache(self, agent_ids, token_boundaries, caches, attention_masks=None):
        """Import already-separated row caches; reject padded/shared boundaries.

        A Transformers batch cache has one physical sequence length, while this
        runner requires each agent's exact logical boundary. The caller must
        therefore provide one cache object and one token boundary per agent.
        """
        if not (len(agent_ids) == len(token_boundaries) == len(caches)):
            raise ValueError("agent_ids, token_boundaries and caches must align")
        if attention_masks is None:
            attention_masks = [None] * len(agent_ids)
        if len(attention_masks) != len(agent_ids):
            raise ValueError("attention_masks must align with agent_ids")
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("each agent must own a distinct cache object")
        prepared = []
        for agent_id, token_ids, cache, mask in zip(agent_ids, token_boundaries, caches, attention_masks):
            if isinstance(cache, (tuple, list)):
                raise ValueError("shared tuple/list batch cache cannot be imported safely")
            if mask is not None and len(mask) != _cache_sequence_length(cache):
                raise ValueError("attention mask length must match imported physical cache length")
            prepared.append((agent_id, save_state(token_ids, cache, mask)))
        return [self.put(agent_id, state) for agent_id, state in prepared]


def longest_common_prefix(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def choose_route(saved: AgentHybridState | None, next_prompt: list[int], last_consumed: int):
    """Return ``reuse`` only when the saved cache covers an exact token prefix."""
    if saved is None:
        return {"route": "fresh", "common_prefix": 0, "suffix_start": 0}
    covered = list(saved.token_ids[:last_consumed])
    common = longest_common_prefix(covered, next_prompt)
    if common != len(covered):
        return {"route": "fresh", "common_prefix": common, "suffix_start": 0}
    return {"route": "reuse", "common_prefix": common, "suffix_start": common}


def _cache_sequence_length(cache):
    getter = getattr(cache, "get_seq_length", None)
    if getter is not None:
        return int(getter())
    layers = getattr(cache, "layers", ())
    for layer in layers:
        keys = getattr(layer, "keys", None)
        if keys is not None:
            return int(keys.shape[-2])
    return None


def save_state(token_ids, cache, cache_attention_mask=None):
    layers = getattr(cache, "layers", ())
    return AgentHybridState(
        token_ids=tuple(token_ids),
        cache=cache,
        conv_states=tuple(getattr(x, "conv_states", None) for x in layers),
        recurrent_states=tuple(getattr(x, "recurrent_states", None) for x in layers),
        kv_cache=tuple((getattr(x, "keys", None), getattr(x, "values", None)) for x in layers),
        cache_attention_mask=None if cache_attention_mask is None else tuple(int(x) for x in cache_attention_mask),
    )


def append_or_fresh(saved, next_prompt, last_consumed, suffix, updated_cache,
                    fresh_cache=None):
    """Route an append and return the route plus the new state.

    The caller performs the actual tokenwise model forwards. This function
    deliberately never crops or pads a state on a prefix mismatch.
    """
    decision = choose_route(saved, next_prompt, last_consumed)
    if decision["route"] == "fresh":
        if fresh_cache is None:
            raise ValueError("fresh_cache is required for a fresh route")
        return decision, save_state(next_prompt, fresh_cache)
    return decision, saved.append(suffix, updated_cache)


class TokenwiseHybridRunner:
    """HF runner adapter that preserves exact single-token GDN semantics."""
    def __init__(self, model):
        self.model = model

    def prefill_agent(self, store: AgentHybridStateStore, agent_id, input_ids,
                      *, attention_mask=None, position_ids=None):
        """Prefill and publish an independent boundary for one agent."""
        state, out = self.prefill(
            input_ids, attention_mask=attention_mask, position_ids=position_ids)
        store.put(agent_id, state)
        return state, out

    def prefill(self, input_ids, *, attention_mask=None, position_ids=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, use_cache=True,
                         return_dict=True, logits_to_keep=1)
        mask = None if attention_mask is None else attention_mask[0].tolist()
        return save_state(input_ids[0].tolist(), out.past_key_values, mask), out

    def append(self, saved, suffix, *, attention_mask, start_position):
        cache = saved.cache
        if saved.cache_attention_mask is not None:
            torch = __import__("torch")
            physical_mask = saved.cache_attention_mask + (1,) * int(suffix.numel())
            attention_mask = torch.tensor([physical_mask], dtype=torch.long, device=suffix.device)
        outputs = []
        for offset, token in enumerate(suffix):
            ids = token.view(1, 1) if hasattr(token, "view") else token
            out = self.model(input_ids=ids, attention_mask=attention_mask,
                             position_ids=ids.new_tensor([[start_position + offset]]),
                             past_key_values=cache, use_cache=True,
                             return_dict=True, logits_to_keep=1)
            cache = out.past_key_values
            outputs.append(out)
        return saved.append(suffix.tolist(), cache), outputs

    def continue_or_fresh(self, saved, next_prompt, last_consumed, *, attention_mask, device):
        decision = choose_route(saved, next_prompt, last_consumed)
        if decision["route"] == "fresh":
            ids = __import__("torch").tensor([next_prompt], device=device)
            fresh, out = self.prefill(ids, attention_mask=attention_mask)
            return decision, fresh, [out]
        suffix = __import__("torch").tensor(
            next_prompt[decision["suffix_start"]:], device=device)
        if suffix.numel() == 0:
            return decision, saved, []
        state, outputs = self.append(saved, suffix, attention_mask=attention_mask,
                                     start_position=decision["suffix_start"])
        return decision, state, outputs

    def continue_agent(self, store: AgentHybridStateStore, agent_id,
                       next_prompt, last_consumed, *, attention_mask, device):
        """Route one agent and atomically publish its resulting boundary."""
        decision, state, outputs = self.continue_or_fresh(
            store.get(agent_id), next_prompt, last_consumed,
            attention_mask=attention_mask, device=device)
        store.put(agent_id, state)
        return decision, state, outputs

    def continue_many(self, store: AgentHybridStateStore, requests, *, device):
        """Schedule independent agents while preserving one tokenwise cache per row.

        ``requests`` contains dictionaries with ``agent_id``, ``next_prompt``,
        ``last_consumed`` and ``attention_mask``. No row is padded into another
        agent's cache; callers may batch the surrounding tool/sampling work.
        """
        results = []
        for request in requests:
            decision, state, outputs = self.continue_agent(
                store, request["agent_id"], request["next_prompt"],
                request["last_consumed"],
                attention_mask=request["attention_mask"], device=device)
            results.append({"agent_id": request["agent_id"],
                            "decision": decision, "state": state,
                            "outputs": outputs})
        return results

    def generate_agent(self, store: AgentHybridStateStore, agent_id,
                       input_ids, *, attention_mask=None, position_ids=None,
                       max_new_tokens=512, eos_token_ids=(), device=None):
        """Greedy-generate one agent while retaining its private cache state."""
        torch = __import__("torch")
        device = device or input_ids.device
        state, out = self.prefill_agent(
            store, agent_id, input_ids, attention_mask=attention_mask,
            position_ids=position_ids)
        generated = []
        eos = set(int(x) for x in (eos_token_ids or ()))
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id in eos:
                break
            position = state.sequence_length
            ids = next_token.view(1, 1).to(device)
            if state.cache_attention_mask is None:
                mask = torch.ones((1, position + 1), dtype=torch.long, device=device)
            else:
                mask = torch.tensor([state.cache_attention_mask + (1,)], dtype=torch.long, device=device)
            state, outputs = self.append(
                state, ids[:, 0], attention_mask=mask,
                start_position=position)
            store.put(agent_id, state)
            next_token = outputs[-1].logits[:, -1, :].argmax(dim=-1)
        return tuple(generated), store.get(agent_id)

    def generate_many(self, store: AgentHybridStateStore, requests, *, device):
        """Upper-layer scheduler for independent greedy agent generations."""
        return [self.generate_agent(
            store, request["agent_id"], request["input_ids"],
            attention_mask=request.get("attention_mask"),
            position_ids=request.get("position_ids"),
            max_new_tokens=request.get("max_new_tokens", 512),
            eos_token_ids=request.get("eos_token_ids", ()), device=device)
            for request in requests]

    def generate_continuation(self, store: AgentHybridStateStore, agent_id,
                              next_prompt, last_consumed, *, attention_mask,
                              max_new_tokens=512, eos_token_ids=(), device=None):
        """Route a new prompt, then greedy-generate from its resulting state."""
        torch = __import__("torch")
        device = device or attention_mask.device
        decision = choose_route(store.get(agent_id), next_prompt, last_consumed)
        if decision["route"] == "fresh":
            ids = torch.tensor([next_prompt], device=device)
            state, out = self.prefill_agent(
                store, agent_id, ids, attention_mask=attention_mask)
        else:
            suffix = torch.tensor(next_prompt[decision["suffix_start"]:], device=device)
            if suffix.numel():
                state, outputs = self.append(
                    store.get(agent_id), suffix,
                    attention_mask=attention_mask,
                    start_position=decision["suffix_start"])
                store.put(agent_id, state)
                out = outputs[-1]
            else:
                state, out = store.get(agent_id), None
        generated = []
        eos = set(int(x) for x in (eos_token_ids or ()))
        if out is None:
            return decision, tuple(generated), state
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        for _ in range(max_new_tokens):
            token_id = int(next_token.item()); generated.append(token_id)
            if token_id in eos: break
            position = state.sequence_length
            if state.cache_attention_mask is None:
                mask = torch.ones((1, position + 1), dtype=torch.long, device=device)
            else:
                mask = torch.tensor([state.cache_attention_mask + (1,)], dtype=torch.long, device=device)
            state, outputs = self.append(
                state, next_token, attention_mask=mask,
                start_position=position)
            store.put(agent_id, state)
            next_token = outputs[-1].logits[:, -1, :].argmax(dim=-1)
        return decision, tuple(generated), store.get(agent_id)

"""Worker hook: replace only the KKT symbol in the original GDN forward."""
import logging
import importlib


def apply_kkt_only_patch():
    from kkt_only import chunk_scaled_dot_kkt_fwd_pto
    for name in (
        'vllm_ascend.ops.triton.fla.chunk',
        'vllm.model_executor.layers.fla.ops.chunk',
    ):
        module = importlib.import_module(name)
        module.chunk_scaled_dot_kkt_fwd = chunk_scaled_dot_kkt_fwd_pto
    logging.getLogger(__name__).warning('PTO KKT-only patch active (C=64)')


# The installed worker hook calls this name. Never enable the staged pipeline.
apply_pto_patch = apply_kkt_only_patch

import torch
from oasis.social_platform import recsys

from app.utils import oasis_recsys_compat


def test_recsys_embeddings_run_without_autograd_in_small_batches(monkeypatch):
    calls = []

    def original(model, tokenizer, texts, batch_size):
        calls.append((torch.is_grad_enabled(), batch_size))
        return "vectors"

    monkeypatch.setattr(oasis_recsys_compat, "_original_generate_post_vector", original)
    monkeypatch.setattr(recsys, "generate_post_vector", recsys.generate_post_vector)

    oasis_recsys_compat.limit_recsys_memory()

    # OASIS asks for one 1000-text batch with autograd on.
    assert recsys.generate_post_vector("model", "tokenizer", ["post"], batch_size=1000) == "vectors"
    assert calls == [(False, oasis_recsys_compat.EMBED_BATCH_SIZE)]

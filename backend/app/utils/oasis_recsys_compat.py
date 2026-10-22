"""
Keep OASIS's TwHIN-BERT recommender within a laptop's memory.

Every round, OASIS re-embeds every agent profile and post in a single batch of up to
1000 texts with autograd enabled, so all BERT layer activations stay alive at once.
With a few hundred agents that alone exceeds 16 GB and the run thrashes swap for
minutes per round. Inference needs no gradients, and BERT's attention mask makes the
embeddings independent of how texts are batched.
"""

import torch
from oasis.social_platform import recsys

# shortcut: wraps an OASIS internal by name; recheck after upgrading camel-oasis.
_original_generate_post_vector = recsys.generate_post_vector
EMBED_BATCH_SIZE = 32


def _generate_post_vector(model, tokenizer, texts, batch_size):
    with torch.no_grad():
        return _original_generate_post_vector(
            model, tokenizer, texts, min(batch_size, EMBED_BATCH_SIZE)
        )


def limit_recsys_memory() -> None:
    recsys.generate_post_vector = _generate_post_vector

"""OneJev fine-tuning (train.sft, unchanged recipe) on 12-band Sentinel-2 media (s1geo.msi).

Adds two options to train.sft (also config keys): `--msi-norm`, the band statistics JSON, and `--input-lr`, the peak
learning rate of the 12-channel input layer (same warmup and cosine schedule as --lr; default: --lr). The model must
be trained with `--full-finetune --multimodal --train-vision`; with s1geo.msi installed, --train-vision trains only
the 12-channel input layer (the rest of the vision tower is frozen as buffers). Ablations: `--msi-train-encoder`
(the whole vision tower is trained) and `--msi-init rgb_mean` (new channels start from the mean RGB kernel).

Run from third_party/OneJev, like train.sft:
    torchrun --standalone --nproc-per-node 1 -m s1geo.msi_sft --config configs/onejev/<msi config>.yaml
"""
from __future__ import annotations

import logging
import sys

log = logging.getLogger("s1geo.msi_sft")


def shard_model_tied(orig):
    """train.fsdp.shard_model for models whose input embedding and LM head share one weight, which OneJev's
    version rejects. Identical otherwise: every decoder layer is an FSDP unit with the same mixed-precision policy; the
    shared embedding/head weight stays in the root unit (gathered once per forward) instead of two units of its own."""
    def shard_model(model, world: int, bf16_compute: bool = True):
        embed, head = model.get_input_embeddings(), model.get_output_embeddings()
        if head is None or head.weight is not embed.weight:
            return orig(model, world, bf16_compute)
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        from train import fsdp

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if bf16_compute else None, reduce_dtype=torch.float32)
        kw = {"mp_policy": mp}
        if dist.get_backend() == "gloo":
            from torch.distributed.device_mesh import DeviceMesh
            kw["mesh"] = DeviceMesh.from_group(dist.group.WORLD, device_type="cuda")

        def to_fp32(module):
            for p in module.parameters(recurse=True):
                if p.is_floating_point() and p.dtype != torch.float32:
                    p.data = p.data.float()

        layers = fsdp.decoder_layers(model)
        for layer in layers:
            to_fp32(layer)
            fully_shard(layer, **kw)
        to_fp32(model)
        fully_shard(model, **kw)
        torch.cuda.empty_cache()
        n = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("FSDP (tied embedding/head kept in the root unit): %d layer units + root over %d ranks, %.2fB parameters "
                 "(%.2fB trained), %.1f GB of fp32 weights+grads+AdamW per rank", len(layers), world, n / 1e9,
                 n_train / 1e9, (16 * n_train + 4 * (n - n_train)) / world / 1e9)
        return model
    return shard_model


def dummy_pass_vision(orig_common):
    """train.fsdp.dummy_pass for a trainable vision input layer (or encoder). OneJev pads a rank that has fewer
    micro-batches with a one-token text-only pass; its vision tower is frozen, so both ranks reduce-scatter the same
    gradients. Here the input layer is trained and lives in the root FSDP unit: a text-only pass gives it no gradient,
    the ranks reduce different sets of tensors and hang. The padding pass therefore adds the trainable vision
    parameters with weight zero: every rank produces a (zero) gradient for the same tensors, and nothing changes."""
    def dummy_pass(model, pad_id: int, device, autocast: bool) -> None:
        import torch

        from s1geo import msi

        ids = torch.tensor([[pad_id]], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = orig_common.readout_logits(model, ids, torch.ones_like(ids), [0])
        vision = [p for p in msi._visual(model).parameters() if p.requires_grad]
        (logits.float().sum() * 0.0 + sum(p.float().sum() for p in vision) * 0.0).backward()
    return dummy_pass


def main(argv: list[str] | None = None) -> int:
    from train import sft

    from s1geo import msi

    orig_build = sft.build_parser

    def build_parser():
        p = orig_build()
        p.add_argument("--msi-norm", default=None, help="band statistics JSON (s1geo.msi.make_norm)")
        p.add_argument("--input-lr", type=float, default=None,
                       help="peak learning rate of the 12-channel input layer (default: --lr)")
        p.add_argument("--msi-init", default="zero", choices=["zero", "rgb_mean"],
                       help="initial weights of the nine new channels (ablation: rgb_mean)")
        p.add_argument("--msi-train-encoder", action="store_true",
                       help="ablation: train the whole vision tower, not only the input layer")
        return p

    sft.build_parser = build_parser
    args = sft.parse_args(argv)
    if not args.msi_norm:
        raise SystemExit("--msi-norm is required")
    if not (args.full_finetune and args.multimodal and args.train_vision):
        raise SystemExit("12-band training needs --full-finetune --multimodal --train-vision "
                         "(--train-vision trains the input layer only)")
    if args.share_prefix:
        raise SystemExit("--share-prefix is not wired for 12-band media")
    msi.install(msi.BandNorm.load(args.msi_norm), freeze=not args.msi_train_encoder, input_lr=args.input_lr,
                init=args.msi_init)
    if args.fsdp:
        from train import fsdp

        from train import common

        fsdp.shard_model = shard_model_tied(fsdp.shard_model)
        fsdp.dummy_pass = dummy_pass_vision(common)
    return sft.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

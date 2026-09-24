import torch
import torch.nn as nn

def _initialize_weights_randomly(self) -> None:
    print("Initializing weights randomly...")
    n_lateral_skipped = 0
    n_preserved = 0
    for name, m in self.named_modules():
        # A module that seeded itself in its block's __init__ (e.g. a Gabor bank
        # on the stage-0 dwconv) marks itself so the from-scratch kaiming below
        # doesn't clobber it — same intent as the "lateral" skip, but by flag
        # since "dwconv" is a name every block has.
        if getattr(m, "_skip_random_init", False):
            n_preserved += 1
            continue
        # The lateral path initializes ITSELF inside the block per the configured
        # init_mode/init_scale — skip it entirely. The LC cube is a custom
        # Parameter (never matched here anyway), but `lateral_pw` IS an nn.Conv2d
        # deliberately seeded dirac_ (identity channel map, sigma1=1). Kaiming
        # fan_out on a (C,C,1,1) conv would silently replace it with a random
        # mixer of sigma1 ~ 2*sqrt(2) ≈ 2.8 → per-step lateral gain ≈ init_scale
        # x 2.8 > 1 → the T-step unroll explodes (per_t loss 1.1 → 54 at batch 0).
        if "lateral" in name:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                n_lateral_skipped += 1
            continue
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if n_lateral_skipped:
        print(f"  (preserved block-owned init on {n_lateral_skipped} lateral module(s), e.g. lateral_pw dirac)")
    if n_preserved:
        print(f"  (preserved block-owned init on {n_preserved} flagged module(s), e.g. stage-0 Gabor dwconv)")
    print("Weight initialization completed.")

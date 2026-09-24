def print_lateral_stats(self, prefix: str = "") -> None:
    """
    Per-layer lateral weight magnitude — call after construction (or at any
    point) to verify the laterals carry signal. Numbers near zero across the
    board mean the lateral term contributes nothing to the forward pass.
    """
    for n, p in self.named_parameters():
        if "lateral" in n:
            print(
                f"{prefix} {n}  shape={tuple(p.shape)}  "
                f"mean|w|={p.abs().mean().item():.4g}  "
                f"max|w|={p.abs().max().item():.4g}"

            )

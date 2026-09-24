def _stage_kernel(stage_idx: int) -> int:
    """cornet_dwsep.Block stage-kernel schedule."""
    return {0: 11, 1: 7, 2: 5, 3: 3, 4: 3}.get(stage_idx, 3)

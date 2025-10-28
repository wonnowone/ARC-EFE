import yaml, torch
from typing import Dict, Any

class FeatureRegistry:
    def __init__(self, cfg_path="configs/operators.yaml"):
        with open(cfg_path,"r") as f:
            self.cfg = yaml.safe_load(f)
        self.ops = {
            "vertical_sym_score": self.vertical_sym_score,
            "axis_color_runlen": self.axis_color_runlen,
            "hole_fill_ratio": self.hole_fill_ratio
        }

    def vertical_sym_score(self, grid: torch.Tensor, **kwargs) -> float:
        flipped = torch.flip(grid, dims=[1])
        return float((grid==flipped).float().mean().item())

    def axis_color_runlen(self, grid: torch.Tensor, axis="columns", color=2, agg="max", **kw) -> float:
        v = 0.0
        if axis=="columns":
            for c in range(grid.shape[1]):
                run, best=0,0
                for r in range(grid.shape[0]):
                    if int(grid[r,c])==int(color): run+=1; best=max(best,run)
                    else: run=0
                v = max(v,best) if agg=="max" else v+best
        return float(v)

    def hole_fill_ratio(self, grid: torch.Tensor, **kw) -> float:

        total = grid.numel()
        bg = (grid==0).sum().item()
        return float(bg/total)

def apply_operator_config(transform_record: Dict[str,Any],
                          input_grid: torch.Tensor,
                          output_grid: torch.Tensor,
                          registry: FeatureRegistry) -> Dict[str,Any]:
    if registry is None: return transform_record
    ops = registry.cfg.get("operators", [])
    for op in ops:
        if not op.get("enabled", True): continue
        name = op["name"]
        fn = registry.ops.get(name)
        if not fn: continue
        params = op.get("params",{})
        try:
            val = fn(input_grid, **params)
            transform_record[name] = val
        except Exception:
            pass
    return transform_record

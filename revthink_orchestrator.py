import torch, json
from typing import Dict, Any, Optional
from qwen_hybrid_prompt import QwenHybridPrompt
from dataclasses import dataclass

@dataclass
class RevThinkCfg:
    tau: float = 0.45    # trigger threshold
    alpha: float = 2.0   # gate sharpness
    beta: float = 0.3    # gate bias
    gamma: float = 0.5   # lambda_prompt boost factor
    eta: float = 0.2     # Z-anchoring blend
    mask_weight: float = 0.5

class RevThinkOrchestrator:
    def __init__(self, qwen: QwenHybridPrompt, cfg: RevThinkCfg):
        self.qwen = qwen
        self.cfg = cfg

    @staticmethod
    def make_issue_report(losses: Dict[str, float], stats: Dict[str, float]) -> Dict[str, Any]:
        keys = ['risk','consistency','bidirectional','ambiguity','total',
                'critique_consistency','tta_consistency','solver_likelihood']
        return {k: float(losses.get(k, stats.get(k, 0.0))) for k in keys}

    def revthink_score(self, losses: Dict[str,float]) -> float:
        w = {'risk':0.2,'consistency':0.2,'bidirectional':0.2,'ambiguity':0.1,
             'critique_consistency':0.15,'tta_consistency':0.1,'solver_likelihood':-0.15}
        s = sum(w[k]*float(losses.get(k,0.0)) for k in w)

        return max(0.0, min(1.0, s))

    @torch.no_grad()
    def maybe_revise(self,
                     transform_record: Dict[str,Any],
                     input_grid: torch.Tensor,
                     output_grid: Optional[torch.Tensor],
                     losses: Dict[str,float]) -> Dict[str,Any]:
        score = self.revthink_score(losses)
        if score < self.cfg.tau:
            return {"apply": False, "revthink_score": score}

        pack = self.qwen(transform_record, input_grid, output_grid, control_weight=0.5)
        audit = pack["audit"]
        confidence = float(audit.get("confidence", 0.6) if isinstance(audit, dict) else 0.6)

        g = torch.sigmoid(torch.tensor(self.cfg.alpha*(confidence*score - self.cfg.beta))).item()

        return {
            "apply": True,
            "revthink_score": score,
            "gate": g,
            "prompt_text": pack["prompt_text"],
            "hybrid_embedding": pack["hybrid_embedding"], 
            "audit": audit
        }

"""
QwenHybridPrompt
- Feature-driven prompt composer (deterministic) + Qwen critique/patch (optional)
- Returns:
    - prompt_text: final short rule sentence (auditable)
    - prompt_embedding: text embedding from Qwen (or fallback)
    - control_vector: numeric-control embedding derived from transformation features
    - hybrid_embedding: fused embedding for ARCPromptGuidedAgent (text⊕control)
    - audit: dict with intermediate artifacts (base/revised, reasons, json parse msg, etc.)

Dependencies:
  pip install transformers accelerate 
"""

from typing import Optional, Dict, Any, Tuple
import json
import os
import re
import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# ----------------------------
# 0) Utilities
# ----------------------------
ARC_COLOR_MAP = {
    0:"black",1:"blue",2:"red",3:"green",4:"yellow",5:"gray",
    6:"magenta",7:"cyan",8:"orange",9:"white"
}

def grid_to_text(grid: torch.Tensor) -> str:
    """Serialize small int grid to compact text with legend."""
    h, w = grid.shape
    rows = [" ".join(str(int(v)) for v in row) for row in grid.cpu().tolist()]
    legend_ids = sorted(set(int(x) for x in grid.unique().tolist()))
    legend = " / ".join([f"{i}:{ARC_COLOR_MAP.get(i,i)}" for i in legend_ids])
    return f"{h}x{w}\n" + "\n".join(rows) + (f"\ncolors: {legend}" if legend else "")

def _clip_sentence(s: str, max_words: int = 25) -> str:
    words = re.findall(r"\S+", s.strip())
    if not words:
        return ""
    if len(words) > max_words:
        words = words[:max_words]
    out = " ".join(words)
    # end with period for stability
    if not out.endswith("."):
        out += "."
    return out

# 1) Feature → base prompt template 

def compose_prompt_from_features(tr: Dict[str, Any]) -> str:
    ttype = tr.get('transformation_type', 'general_transformation')
    size_ratio = float(tr.get('size_change_ratio', 1.0))
    colors_added = int(tr.get('colors_added', 0) or 0)
    colors_removed = int(tr.get('colors_removed', 0) or 0)
    sym = float(tr.get('symmetry_change', 0.0))
    dens = float(tr.get('density_change', 0.0))
    corr = float(tr.get('spatial_correlation', 0.5))

    base = {
        'shrinking': 'shrink the grid while preserving pattern structure',
        'expanding': 'expand the grid by repeating local pattern',
        'reshaping': 'reshape grid while keeping relative positions',
        'size_transformation': 'change grid size and structure consistently',
        'copy_or_minimal': 'copy input with minimal edits',
        'recoloring': 'recolor pattern; keep spatial structure',
        'color_addition': 'add new colors to the pattern',
        'color_removal': 'remove specific colors from pattern',
        'symmetry_operation': 'apply symmetry (mirror/rotate) to align pattern',
        'spatial_shift': 'translate pattern without changing structure',
        'pattern_completion': 'complete missing pattern elements',
        'pattern_removal': 'remove redundant pattern elements',
        'reconstruction': 'reconstruct pattern with transformation rules',
        'filtering': 'filter pattern elements by rule',
        'pattern_transformation': 'transform the pattern structure systematically',
        'general_transformation': 'apply systematic transformation to the pattern',
    }.get(ttype, 'apply systematic transformation to the pattern')

    parts = [base]
    if size_ratio < 0.9: parts.append(f"target size ≈ {size_ratio:.0%} of input")
    elif size_ratio > 1.1: parts.append(f"target size ≈ {size_ratio:.0%} of input")
    if colors_added > 0: parts.append(f"introduce {colors_added} new colors")
    if colors_removed > 0: parts.append(f"remove {colors_removed} colors")
    if sym > 0.5: parts.append("consider symmetry change")
    if dens > 0.2: parts.append("become denser")
    elif dens < -0.2: parts.append("become sparser")
    if corr > 0.7: parts.append("preserve strong spatial relations")
    elif corr < 0.3: parts.append("allow large reorganization")

    return _clip_sentence(". ".join(parts) + ".")

# 2) Feature → control vector (numeric)

class FeatureToControl(nn.Module):
    """Map numeric transformation features to agent prompt space."""
    def __init__(self, in_dim: int = 15, prompt_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, prompt_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def features_to_numeric_vec(tr: Dict[str, Any]) -> torch.Tensor:
    vals = [
        float(tr.get('size_change_ratio',1.0)),
        float(tr.get('size_preserved',0.0)),
        float(tr.get('pixel_change_ratio',1.0)),
        float(tr.get('color_preservation',0.0)),
        float(tr.get('spatial_correlation',0.0)),
        float(tr.get('symmetry_change',0.0)),
        float(tr.get('density_change',0.0)),
        float(tr.get('colors_added',0)),
        float(tr.get('colors_removed',0)),
        float(tr.get('input_color_count',0)),
        float(tr.get('output_color_count',0)),
        float(tr.get('input_height',0)),
        float(tr.get('input_width',0)),
        float(tr.get('output_height',0)),
        float(tr.get('output_width',0)),
    ]
    return torch.tensor(vals, dtype=torch.float32)

# Qwen critique & embedding

QWEN_SYSTEM = (
    "You are an ARC puzzle rule auditor. "
    "You will receive input and optionally output grids and a candidate one-sentence rule. "
    "Return a STRICT JSON with keys: {\"keep\": bool, \"reason\": str, \"revised\": str}."
)

QWEN_PROMPT = """InputGrid:
{in_txt}

{maybe_out}CandidateRule: "{candidate}"

Constraints:
- If rule is already precise, set keep=true and revised=candidate.
- If not, set keep=false and provide a corrected concise rule (<= 25 words).
- Prefer terms: mirror/rotate/translate/recolor/fill/scale/complete/remove/pattern.
- Mention color index changes like "2→1" only if clearly implied.

JSON only:
"""

def _make_qwen_prompt(input_grid: torch.Tensor,
                      output_grid: Optional[torch.Tensor],
                      candidate: str) -> str:
    in_txt = grid_to_text(input_grid)
    if output_grid is not None:
        out_txt = "OutputGrid:\n" + grid_to_text(output_grid) + "\n\n"
    else:
        out_txt = ""
    return QWEN_PROMPT.format(in_txt=in_txt, maybe_out=out_txt, candidate=candidate)

def _safe_json_parse(s: str) -> Tuple[Optional[Dict[str, Any]], str]:
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None, "no-json"
    block = m.group(0)
    try:
        obj = json.loads(block)
        keep = bool(obj.get("keep", False))
        reason = str(obj.get("reason", ""))
        revised = str(obj.get("revised", "")).strip()
        return {"keep": keep, "reason": reason, "revised": revised}, "ok"
    except Exception as e:
        return None, f"json-error: {e}"


# 4) Prompt cache
class PromptCache:
    def __init__(self, path: str = ".cache/qwen_hybrid"):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def _key(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor], candidate: str) -> str:
        payload = {
            "in": input_grid.cpu().tolist(),
            "out": (output_grid.cpu().tolist() if output_grid is not None else None),
            "cand": candidate
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def load(self, input_grid, output_grid, candidate):
        k = self._key(input_grid, output_grid, candidate)
        f = os.path.join(self.path, k + ".json")
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as h:
                return json.load(h)
        return None

    def save(self, input_grid, output_grid, candidate, data):
        k = self._key(input_grid, output_grid, candidate)
        f = os.path.join(self.path, k + ".json")
        with open(f, "w", encoding="utf-8") as h:
            json.dump(data, h, ensure_ascii=False, indent=2)


# ----------------------------
# 5) Main hybrid class
# ----------------------------
@dataclass
class QwenCfg:
    model_name: str = "Qwen/Qwen2.5-1.8B"
    dtype: str = "float16"          # "float16" | "bfloat16"
    max_new_tokens: int = 96
    temperature: float = 0.0        # deterministic by default
    top_p: float = 1.0
    embed_pool: str = "mean"        # "mean" | "cls"
    cache_dir: Optional[str] = ".cache/hf"
    use_qwen: bool = True           # turn off to skip any Qwen calls

class QwenHybridPrompt(nn.Module):
    """
    - Compose deterministic base rule from features
    - (Optional) Ask Qwen to audit/patch
    - Produce text embedding (Qwen) and control vector (numeric MLP)
    - Fuse to hybrid embedding for agent
    """
    def __init__(self,
                 prompt_dim: int = 256,
                 numeric_in_dim: int = 15,
                 fuse: str = "mean",              
                 fuse_proj_dim: Optional[int] = None,
                 qwen: Optional[QwenCfg] = None,
                 cache_prompts: bool = True):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.ctrl_mapper = FeatureToControl(in_dim=numeric_in_dim, prompt_dim=prompt_dim)
        self.fuse = fuse
        self.cache = PromptCache() if cache_prompts else None

        self._use_qwen = bool(qwen.use_qwen) if qwen else False
        self._embed_pool = (qwen.embed_pool if qwen else "mean").lower()

        if fuse == "concat-linear":
            if fuse_proj_dim is None:
                fuse_proj_dim = prompt_dim
            self.fuse_proj = nn.Linear(prompt_dim * 2, fuse_proj_dim)
            self.hybrid_dim = fuse_proj_dim
        else:
            self.hybrid_dim = prompt_dim  # mean-fuse keeps size

        # Lazy-load transformers
        self._has_tf = _HAS_TRANSFORMERS and self._use_qwen
        if self._has_tf:
            dtype = torch.float16 if (qwen and qwen.dtype.lower()=="float16") else torch.bfloat16
            name = qwen.model_name
            cache_dir = qwen.cache_dir
            self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
            self.lm = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, device_map="auto", cache_dir=cache_dir
            )
            self.embedder = AutoModel.from_pretrained(
                name, torch_dtype=dtype, device_map="auto", cache_dir=cache_dir
            )
            self.max_new = int(qwen.max_new_tokens)
            self.temp = float(qwen.temperature)
            self.top_p = float(qwen.top_p)

    # ---------- public API ----------
    @torch.no_grad()
    def forward(self,
                transform_record: Dict[str, Any],
                input_grid: torch.Tensor,
                output_grid: Optional[torch.Tensor] = None,
                control_weight: float = 0.5) -> Dict[str, Any]:
        """
        Returns dict:
          - prompt_text
          - prompt_embedding       [prompt_dim or embed_dim from Qwen]
          - control_vector         [prompt_dim]
          - hybrid_embedding       [hybrid_dim]
          - audit: {...}
        """
        device = next(self.ctrl_mapper.parameters()).device

        # 1) base rule from features
        base_rule = compose_prompt_from_features(transform_record)

        audit = {
            "base_rule": base_rule,
            "qwen_used": bool(self._has_tf),
            "qwen_parse": None,
            "qwen_reason": None,
            "revised_rule": None
        }

        # Qwen critique/patch
        final_rule = base_rule
        if self._has_tf:
            cached = self.cache.load(input_grid, output_grid, base_rule) if self.cache else None
            if cached is not None:
                parsed = cached
                audit["qwen_parse"] = "cache"
            else:
                sys = QWEN_SYSTEM
                prompt = _make_qwen_prompt(input_grid, output_grid, base_rule)
                inputs = self.tokenizer(
                    sys + "\n\n" + prompt,
                    return_tensors="pt"
                ).to(self.lm.device)
                gen = self.lm.generate(
                    **inputs, max_new_tokens=self.max_new, do_sample=(self.temp>0),
                    temperature=self.temp, top_p=self.top_p,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
                parsed, status = _safe_json_parse(text)
                audit["qwen_parse"] = status
                if self.cache:
                    self.cache.save(input_grid, output_grid, base_rule, parsed if parsed else {"raw": text, "status": status})

            if parsed:
                keep = bool(parsed.get("keep", False))
                reason = str(parsed.get("reason", ""))
                revised = _clip_sentence(parsed.get("revised", "") or base_rule)
                audit["qwen_reason"] = reason
                audit["revised_rule"] = revised
                final_rule = base_rule if keep else revised

        # 3) text embedding 
        if self._has_tf:
            tokens = self.tokenizer(final_rule, return_tensors="pt").to(self.embedder.device)
            out = self.embedder(**tokens)
            hs = out.last_hidden_state  
            if self._embed_pool == "cls" and hs.shape[1] > 0:
                text_emb = hs[:, 0, :]
            else:
                text_emb = hs.mean(dim=1)
            text_emb = text_emb.squeeze(0).float()
        else:
            # fallback: zero vector
            text_emb = torch.zeros(self.prompt_dim)

        # 4) control vector from numeric features
        num_vec = features_to_numeric_vec(transform_record).to(device)
        ctrl_vec = self.ctrl_mapper(num_vec)  

        # 5) fuse → hybrid embedding (for agent)
        if self._has_tf:
            # project (if dims mismatch) to prompt_dim for fusion
            if text_emb.dim() == 1:
                # attempt to match ctrl_vec dim; if not equal, simple linear projection
                if text_emb.shape[-1] != ctrl_vec.shape[-1]:
                    proj = nn.Linear(text_emb.shape[-1], ctrl_vec.shape[-1]).to(text_emb.device)
                    with torch.no_grad():
                        nn.init.xavier_uniform_(proj.weight)
                        nn.init.zeros_(proj.bias)
                    text_emb = proj(text_emb)
            text_emb = text_emb.to(ctrl_vec.device)
        else:
            text_emb = text_emb.to(ctrl_vec.device)

        if self.fuse == "concat-linear":
            fused = torch.cat([ctrl_vec, text_emb], dim=-1)
            hybrid = self.fuse_proj(fused)
        else:
            # mean-fuse with weight to keep control influence
            hybrid = (1.0 - control_weight) * text_emb + control_weight * ctrl_vec

        return {
            "prompt_text": final_rule,
            "prompt_embedding": text_emb.detach().cpu(),   
            "control_vector": ctrl_vec.detach().cpu(),
            "hybrid_embedding": hybrid.detach(),          
            "audit": audit
        }

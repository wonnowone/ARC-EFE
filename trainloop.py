import torch, os, time, json
from torch.utils.data import DataLoader
from typing import Dict, Any
from dataset_arc import ARCDataset
from qwen_hybrid_prompt import QwenHybridPrompt, QwenCfg
from revthink_orchestrator import RevThinkOrchestrator, RevThinkCfg
from loss_function import ARCPromptGuidedAgent
from solver1 import ContextualSolver 
from solver2 import PermanentSolver  
from tta import TestTimeAdaptationSystem      

from feature_registry import FeatureRegistry, apply_operator_config
from feature_extraction import extract_transformation_features, classify_transformation_type

# Optional: TTA monitoring config
TTA_EVAL_INTERVAL = 50  # Evaluate TTA every N batches (set to 0 to disable)

def seed_all(seed=42):
    import random, numpy as np
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

def make_agent(device="cuda"):
    agent = ARCPromptGuidedAgent(max_grid_size=30, num_colors=10, hidden_dim=256, prompt_dim=256).to(device)
    return agent

def make_qwen(device="cuda"):
    qcfg = QwenCfg(model_name="Qwen/Qwen2.5-1.5B", dtype="float16", temperature=0.0, use_qwen=True)
    qwen = QwenHybridPrompt(prompt_dim=256, numeric_in_dim=15, fuse="mean", qwen=qcfg).to(device)
    return qwen

def pack_transform_record(inp, out):
    # Squeeze batch dimension if present (input comes from batch loader)
    if inp.dim() == 3:
        inp = inp.squeeze(0)
    if out is not None and out.dim() == 3:
        out = out.squeeze(0)

    # numpy list → python list → dict
    in_grid = inp.cpu().tolist()
    out_grid = out.cpu().tolist() if out is not None else None
    if out_grid is None:
        # test split: output 없음 → dummy로 동일 크기 zeros
        out_grid = [[0 for _ in row] for row in in_grid]
    feats = extract_transformation_features(in_grid, out_grid)
    feats["transformation_type"] = classify_transformation_type(feats)
    return feats

def train_one_epoch(agent, loader, optim, device, qwen, revthink, results_sink, feat_reg, tta_system=None, epoch=0):
    agent.train()
    qwen.train()  # Enable training mode for Qwen (batch norm, dropout, etc)

    # Store original lambda_prompt to prevent unbounded accumulation
    original_lambda_prompt = agent.efe_loss.lambda_prompt

    for batch_idx, batch in enumerate(loader):
        inp = batch["input"].to(device)
        out = batch["output"].to(device) if batch["output"] is not None else None
        prob_id, idx = batch["prob_id"], batch["idx"]

        # 1) feature → transform_record
        tr = pack_transform_record(inp, out)

        # 2) feature_registry 적용(연산자 config에 따른 추가 피처)
        tr = apply_operator_config(tr, inp, out, feat_reg)

        # 3) 하이브리드 프롬프트 생성(Qwen 심사/패치 포함)
        with torch.no_grad():
            pack = qwen(tr, inp, out, control_weight=0.5)
        prompt_text = pack["prompt_text"]
        prompt_emb = pack["hybrid_embedding"].squeeze(0)

        # 4) 에피소드 학습
        optim.zero_grad()
        losses = agent.train_episode(
            initial_state=inp, target_state=out,
            prompt_text=prompt_text, prompt_embedding=prompt_emb,
            num_steps=3
        )

        # 5) RevThink: 손실 기반 개입(있으면)
        losses_num = {k: (v.item() if hasattr(v, "item") else v) for k, v in losses.items() if k!="prompt_text"}
        issue = revthink.maybe_revise(tr, inp, out, losses_num)
        if issue["apply"]:
            g = issue["gate"]
            new_emb = (1-g)*prompt_emb + g*issue["hybrid_embedding"].squeeze(0)
            # λ_prompt 강화 (reset to original then apply boost to prevent unbounded growth)
            agent.efe_loss.lambda_prompt = original_lambda_prompt * (1 + revthink.cfg.gamma * g)
            # 재계산(선택): 비용 적게 하려면 다음 step에 반영
            losses = agent.train_episode(inp, out, issue["prompt_text"], new_emb, num_steps=3)
        else:
            # Reset to original if no revision needed
            agent.efe_loss.lambda_prompt = original_lambda_prompt

        # 6) 역전파
        losses["total"].backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

        optim.step()

        # 7) 결과 저장
        results_sink.write_train(prob_id, int(idx), losses, prompt_text)

        # 8) Optional: Periodic TTA evaluation
        if TTA_EVAL_INTERVAL > 0 and tta_system is not None and batch_idx % TTA_EVAL_INTERVAL == 0:
            evaluate_tta(tta_system, batch, device, results_sink, epoch, batch_idx)

def evaluate_tta(tta_system, batch, device, results_sink, epoch, batch_idx):
    """Periodically evaluate TTA system on a sample batch."""
    try:
        inp = batch["input"].to(device)
        out = batch["output"].to(device) if batch["output"] is not None else None
        prob_id = batch["prob_id"]

        # Create simple prompt embedding for TTA eval
        prompt_emb = torch.randn(256, device=device)
        prompt_text = "TTA evaluation"

        results = tta_system.test_time_adapt(inp, out, prompt_text, prompt_emb)

        # Log TTA metrics
        tta_metrics = {
            "epoch": epoch,
            "batch": batch_idx,
            "prob_id": prob_id,
            "selected_solver": results['selected_solver'],
            "final_surprise": results['final_surprise'],
            "memory_size": results['memory_size'],
            "avg_adaptation_loss": sum(results['adaptation_losses']) / max(len(results['adaptation_losses']), 1)
        }

        tta_log_path = os.path.join(results_sink.outdir, "tta_eval.jsonl")
        with open(tta_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(tta_metrics, ensure_ascii=False) + "\n")

    except Exception as e:
        # Log but don't crash training on TTA eval failure
        print(f"[Warning] TTA eval failed at epoch {epoch} batch {batch_idx}: {str(e)[:50]}")

def evaluate(agent, loader, device, qwen, feat_reg, results_sink):
    agent.eval()
    qwen.eval()  # Disable training mode for Qwen during evaluation
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            out = batch["output"].to(device) if batch["output"] is not None else None
            prob_id, idx = batch["prob_id"], batch["idx"]
            tr = pack_transform_record(inp, out)
            tr = apply_operator_config(tr, inp, out, feat_reg)
            pack = qwen(tr, inp, out, control_weight=0.5)
            preds, _ = agent.forward_planning(inp, pack["hybrid_embedding"].squeeze(0), num_steps=5)
            final = preds[-1].argmax(dim=-1)
            if out is not None:
                acc = (final == out).float().mean().item()
                acc_sum += acc; n += 1
                results_sink.write_eval(prob_id, int(idx), acc, pack["prompt_text"])
    return acc_sum / max(n,1)

def main():
    seed_all()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ARCDataset("training.json", split="train")
    test_ds  = ARCDataset("training.json", split="test")
    train_ld = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False)

    agent = make_agent(device)
    qwen = make_qwen(device)
    revthink = RevThinkOrchestrator(qwen=qwen, cfg=RevThinkCfg())
    feat_reg = FeatureRegistry()   # 연산자 레지스트리

    # Create TTA system for optional monitoring (can disable by setting TTA_EVAL_INTERVAL = 0)
    tta_system = TestTimeAdaptationSystem(
        base_agent=agent,
        memory_size=500,
        num_solvers=3,
        adaptation_steps=3,
        adaptation_lr=1e-3
    )

    optim = torch.optim.Adam(agent.parameters(), lr=1e-3)
    results_sink = ResultsSink(outdir="runs/exp1")

    best, ckpt = -1, os.path.join(results_sink.outdir, "agent_best.pt")
    for epoch in range(5):
        train_one_epoch(agent, train_ld, optim, device, qwen, revthink, results_sink, feat_reg, tta_system, epoch)
        acc = evaluate(agent, test_ld, device, qwen, feat_reg, results_sink)
        results_sink.log_metric("epoch_acc", epoch, acc)
        print(f"[Epoch {epoch}] test_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(agent.state_dict(), ckpt)

class ResultsSink:
    def __init__(self, outdir="runs/exp1"):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.train_log = open(os.path.join(outdir, "train_log.jsonl"), "a", encoding="utf-8")
        self.eval_log  = open(os.path.join(outdir, "eval_log.jsonl"), "a", encoding="utf-8")

    def write_train(self, prob_id, idx, losses: Dict[str,Any], prompt_text:str):
        rec = {"prob_id":prob_id,"idx":idx,"prompt":prompt_text}
        for k,v in losses.items():
            if k=="prompt_text": continue
            rec[k] = (v.item() if hasattr(v,"item") else v)
        self.train_log.write(json.dumps(rec, ensure_ascii=False)+"\n"); self.train_log.flush()

    def write_eval(self, prob_id, idx, acc, prompt_text):
        rec = {"prob_id":prob_id,"idx":idx,"acc":acc,"prompt":prompt_text}
        self.eval_log.write(json.dumps(rec, ensure_ascii=False)+"\n"); self.eval_log.flush()

    def log_metric(self, name, step, value):
        path = os.path.join(self.outdir, "metrics.jsonl")
        with open(path,"a",encoding="utf-8") as f:
            f.write(json.dumps({"name":name,"step":step,"value":value})+"\n")

if __name__ == "__main__":
    main()

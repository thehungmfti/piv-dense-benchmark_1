"""Huấn luyện RAFT-PIV (tái hiện) — vòng lặp CPU/GPU đơn giản, có checkpoint & tiếp tục."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .model import RAFTPIV, RAFTPIVConfig, count_parameters, sequence_loss


def save_checkpoint(path, model, cfg: RAFTPIVConfig, step: int, opt=None, sched=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model=model.state_dict(), config=cfg.to_dict(), step=step, extra=extra or {})
    if opt is not None:
        state["optimizer"] = opt.state_dict()
    if sched is not None:
        state["scheduler"] = sched.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_model(path, device="cpu") -> tuple[RAFTPIV, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg_d = dict(state["config"])
    cfg_d["enc_channels"] = tuple(cfg_d["enc_channels"])
    model = RAFTPIV(RAFTPIVConfig(**cfg_d))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, state


@torch.no_grad()
def evaluate_epe(model, samples, iters: int = 12, device="cpu", border: int = 8) -> float:
    """EPE trung bình (bỏ biên, chỉ điểm hợp lệ) trên danh sách (img1, img2, flow(2,H,W), valid(H,W))."""
    from .infer import predict

    model.eval()
    errs = []
    for a, b, fl, va in samples:
        u, v = predict(model, a, b, iters=iters, device=device)
        e = np.hypot(u - fl[0], v - fl[1])
        m = np.asarray(va, bool).copy()
        if border:
            m[:border], m[-border:], m[:, :border], m[:, -border:] = False, False, False, False
        errs.append(float(e[m].mean()))
    model.train()
    return float(np.mean(errs))


def train(cfg_model: RAFTPIVConfig, dataset, out_dir, steps: int = 6000, batch: int = 6, lr: float = 4e-4,
          weight_decay: float = 1e-5, gamma: float = 0.8, clip: float = 1.0, val_sets: dict | None = None,
          val_every: int = 500, ckpt_every: int = 250, threads: int | None = None, resume: bool = True,
          device: str = "cpu", time_limit_h: float | None = None, log_every: int = 25, init_from=None):
    """``init_from``: checkpoint khởi tạo trọng số (tinh chỉnh) khi chưa có ``last.pt`` trong ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = RAFTPIV(cfg_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps + 100, pct_start=0.05,
                                                cycle_momentum=False, anneal_strategy="linear")
    step = 0
    last = out / "last.pt"
    if resume and last.is_file():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        if "optimizer" in st:
            opt.load_state_dict(st["optimizer"])
        if "scheduler" in st:
            sched.load_state_dict(st["scheduler"])
        step = int(st["step"])
        print(f"Tiếp tục từ bước {step}", flush=True)
    elif init_from:
        st = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        print(f"Khởi tạo trọng số từ {init_from} (bước {st.get('step')})", flush=True)
    log = open(out / "train_log.jsonl", "a")
    print(f"RAFT-PIV: {count_parameters(model) / 1e6:.2f} M tham số, cấu hình {cfg_model.to_dict()}", flush=True)
    best = math.inf
    best_file = out / "best.json"
    if best_file.is_file():
        best = json.loads(best_file.read_text())["val"]
    model.train()
    t0 = time.time()
    run_loss, run_epe, n_run = 0.0, 0.0, 0
    while step < steps:
        a, b, fl, va = dataset.batch(batch)
        a, b = torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
        fl, va = torch.from_numpy(fl).to(device), torch.from_numpy(va).to(device)
        preds = model(a, b, iters=cfg_model.iters)
        loss, epe = sequence_loss(preds, fl, va, gamma=gamma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        step += 1
        run_loss += float(loss.detach())
        run_epe += float(epe.detach())
        n_run += 1
        if step % log_every == 0:
            rec = dict(step=step, loss=run_loss / n_run, epe=run_epe / n_run, lr=sched.get_last_lr()[0],
                       elapsed_s=time.time() - t0)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step:6d} loss {rec['loss']:.4f} epe {rec['epe']:.4f} lr {rec['lr']:.2e} "
                  f"{rec['elapsed_s'] / 60:.1f} min", flush=True)
            run_loss, run_epe, n_run = 0.0, 0.0, 0
        if step % ckpt_every == 0 or step == steps:
            save_checkpoint(last, model, cfg_model, step, opt, sched)
        if val_sets and (step % val_every == 0 or step == steps):
            res = {k: evaluate_epe(model, v, iters=cfg_model.iters + 4, device=device) for k, v in val_sets.items()}
            score = float(np.mean(list(res.values())))
            log.write(json.dumps(dict(step=step, val=res, val_mean=score)) + "\n")
            log.flush()
            print(f"  [val] step {step}: {res} (mean {score:.4f})", flush=True)
            if score < best:
                best = score
                save_checkpoint(out / "best.pt", model, cfg_model, step, extra=dict(val=res))
                best_file.write_text(json.dumps(dict(step=step, val=score, detail=res)))
        if time_limit_h and (time.time() - t0) > time_limit_h * 3600:
            save_checkpoint(last, model, cfg_model, step, opt, sched)
            print(f"Dừng do giới hạn thời gian tại bước {step}", flush=True)
            break
    log.close()
    return model

from __future__ import annotations
"""
PoC: Turning-point candidates from event JSON.

- Build 3D field rho(t) from A/B/C scores
- Forward scattering via kimura2 PhysicsForwardModel (loaded from src/kimura2.txt)
- 4D reconstruction with dynamic temporal regularization:
    * resource C low -> mu up (stabilize)
    * contradiction high -> mu down a bit (keep "invention points")
- Rank events by (warp * contradiction)

Run:
  python -m src.poc_turning_points --json data/sample/sample_g_1_reconstructed_keep_last9.json
"""

import argparse
import json
import re
import types
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn


def load_kimura2_from_txt(path: str):
    """Load kimura2.txt safely by removing demo/benchmark tail (if any) before exec."""
    src = open(path, "r", encoding="utf-8").read()
    cut_patterns = [
        r'^\s*print\("Generating EMF Synthetic Data',
        r'^\s*forward\s*=\s*PhysicsForwardModel',
        r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:',
    ]
    lines = src.splitlines(True)
    cut_idx = None
    for i, line in enumerate(lines):
        if any(re.search(p, line) for p in cut_patterns):
            cut_idx = i
            break
    safe_src = "".join(lines[:cut_idx]) if cut_idx is not None else src
    mod = types.ModuleType("kimura2_safe")
    exec(safe_src, mod.__dict__)
    return mod


@dataclass
class EventRec:
    chapter_title: str
    scene_id: str
    global_step: int
    local_time: float
    event: str
    desc: str
    evidence: List[str]
    status: str
    label: int | None

    A_approach: float = 0.5
    B_threat: float = 0.5
    C_resource: float = 0.5


def load_events(json_path: str) -> List[EventRec]:
    data = json.loads(open(json_path, "r", encoding="utf-8").read())
    out: List[EventRec] = []
    for ch in data:
        chapter_title = ch["chapter"]["title"]
        scene_id = ch["chapter"]["scene_id"]
        for r in ch["time_series_data"]:
            out.append(
                EventRec(
                    chapter_title=chapter_title,
                    scene_id=scene_id,
                    global_step=int(r["global_step"]),
                    local_time=float(r.get("local_time", 0.0)),
                    event=str(r.get("event", "")),
                    desc=str(r.get("desc", "")),
                    evidence=[str(x) for x in r.get("evidence", [])],
                    status=str(r.get("status", "unlabeled")),
                    label=(int(r["label"]) if r.get("status") == "labeled" and "label" in r else None),
                )
            )
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def keyword_score(text: str, pos: List[str], neg: List[str]) -> float:
    s = 0.0
    for w in pos:
        if w in text:
            s += 1.0
    for w in neg:
        if w in text:
            s -= 1.0
    return s


# ---- simple Japanese heuristics for PoC
KW_APPROACH = ["近づ", "話", "一緒", "仲間", "助け", "頼", "笑", "共有", "参加", "手を上げ"]
KW_AVOID    = ["逃", "隠", "目をそら", "引っ込", "離", "黙", "避け", "閉じこも", "消えたい"]
KW_THREAT   = ["からか", "嘲", "恥", "怖", "怒", "叱", "指名", "注目", "失敗", "排除", "投げ"]
KW_SAFE     = ["安心", "優し", "守", "受け入", "支え", "落ち着", "信頼", "味方", "大丈夫"]
KW_DEPLETE  = ["疲", "眠", "だる", "金が", "空腹", "余裕がない", "限界", "しんどい"]
KW_ABUND    = ["元気", "回復", "余裕", "満た", "安心できる", "落ち着ける"]


def score_axes(e: EventRec, scale: float = 1.8) -> EventRec:
    text = " / ".join([e.event, e.desc, " ".join(e.evidence)])

    rawA = keyword_score(text, KW_APPROACH, KW_AVOID)
    rawB = keyword_score(text, KW_THREAT, KW_SAFE)
    rawC = keyword_score(text, KW_ABUND, KW_DEPLETE)

    if e.label is not None:
        if e.label == 1:
            rawB += 0.8; rawC -= 0.4; rawA -= 0.2
        elif e.label == 0:
            rawB -= 0.6; rawC += 0.4; rawA += 0.2

    e.A_approach = float(sigmoid(rawA / scale))
    e.B_threat   = float(sigmoid(rawB / scale))
    e.C_resource = float(sigmoid(rawC / scale))
    return e


def time_bin(local_time: float, T: int) -> int:
    t = float(np.clip(local_time, 0.0, 1.0))
    b = int(np.floor(t * T))
    return min(b, T - 1)


def nan_fill_fb(x: np.ndarray, default=0.5):
    x = x.copy()
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = x[i-1] if i > 0 else np.nan
    for i in range(len(x)-1, -1, -1):
        if np.isnan(x[i]):
            x[i] = x[i+1] if i < len(x)-1 else default
    x[np.isnan(x)] = default
    return x


def contradiction_score(e: EventRec) -> float:
    return float(e.A_approach * e.B_threat * (1.0 - e.C_resource))


def frame_series(events: List[EventRec], T: int):
    binsC = [[] for _ in range(T)]
    binsK = [[] for _ in range(T)]
    for e in events:
        t = time_bin(e.local_time, T)
        binsC[t].append(e.C_resource)
        binsK[t].append(contradiction_score(e))

    resource = np.array([np.mean(b) if len(b) else np.nan for b in binsC], dtype=np.float64)
    resource = np.clip(nan_fill_fb(resource, default=0.5), 0, 1)

    contr = np.empty(T, dtype=np.float64)
    for t in range(T):
        arr = np.array(binsK[t], dtype=np.float64)
        if arr.size == 0:
            contr[t] = np.nan
        elif arr.size >= 5:
            thr = np.quantile(arr, 0.90)
            contr[t] = float(np.mean(arr[arr >= thr]))
        else:
            contr[t] = float(np.mean(arr))
    contr = np.clip(nan_fill_fb(contr, default=0.0), 0, 1)
    return resource, contr


def build_density_grid(events: List[EventRec], N: int = 48, sigma: float = 1.1):
    hits = np.zeros((N, N, N), dtype=np.float32)
    for e in events:
        x = int(round(e.A_approach * (N - 1)))
        y = int(round(e.B_threat   * (N - 1)))
        z = int(round(e.C_resource * (N - 1)))
        w = 1.0
        if e.label == 1: w = 1.2
        if e.label == 0: w = 0.9
        hits[z, y, x] += w

    rho = gaussian_filter(hits, sigma=sigma)
    if rho.max() > 0:
        rho = rho / rho.max()
    return rho


def build_rho_sequence(events: List[EventRec], T: int, N: int):
    rho_seq = np.zeros((T, N, N, N), dtype=np.float32)
    counts = np.zeros(T, dtype=np.int32)
    for t in range(T):
        ev_t = [e for e in events if time_bin(e.local_time, T) == t]
        counts[t] = len(ev_t)
        if len(ev_t) > 0:
            rho_seq[t] = build_density_grid(ev_t, N=N, sigma=1.1)
    return rho_seq, counts


def make_mu_edges(resource_t, contr_t, mu_min=0.01, mu_max=0.12, power=1.0, alpha=0.6, mu_floor=0.006):
    T = len(resource_t)
    r_edge = 0.5 * (resource_t[:-1] + resource_t[1:])
    c_edge = 0.5 * (contr_t[:-1] + contr_t[1:])

    scarcity = (1.0 - r_edge) ** power
    mu_base = mu_min + (mu_max - mu_min) * scarcity
    factor = 1.0 - alpha * np.clip(c_edge, 0.0, 1.0)
    mu = np.maximum(mu_base * factor, mu_floor)
    return mu, mu_base, factor


def thomas_batch(l, a, u, d):
    T, M = a.shape
    cp = np.zeros((T, M), dtype=np.complex128)
    dp = np.zeros((T, M), dtype=np.complex128)

    denom = a[0]
    cp[0] = u[0] / denom
    dp[0] = d[0] / denom

    for i in range(1, T):
        denom = a[i] - l[i] * cp[i-1]
        cp[i] = (u[i] / denom) if i < T-1 else 0.0
        dp[i] = (d[i] - l[i] * dp[i-1]) / denom

    x = np.zeros((T, M), dtype=np.complex128)
    x[T-1] = dp[T-1]
    for i in range(T-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x


class KimuraSolver4DDynamic:
    def __init__(self, forward_model, size=48, reg_lambda=1e-3, pad_factor=2, normalize="global"):
        self.forward = forward_model
        self.size = int(size)
        self.reg_lambda = float(reg_lambda)
        self.pad_factor = int(pad_factor)
        self.normalize = normalize

    def reconstruct_sequence(self, G_seq, mu_edges):
        G_seq = np.asarray(G_seq)
        T, N, _, _ = G_seq.shape
        assert N == self.size
        mu_edges = np.asarray(mu_edges, dtype=np.float64)
        assert mu_edges.shape == (T-1,)

        N_pad = self.pad_factor * N
        H = self.forward._fft_green(N_pad)
        Hf = H.reshape(-1)
        absH2 = (np.abs(Hf) ** 2).astype(np.float64)

        Y = np.zeros((T, Hf.size), dtype=np.complex128)
        for t in range(T):
            Yp = self.forward._pad_center(G_seq[t].astype(np.complex128), N_pad)
            Y[t] = fftn(Yp).reshape(-1)

        d = np.conj(Hf)[None, :] * Y

        lam = self.reg_lambda
        left = np.zeros(T, dtype=np.float64)
        right = np.zeros(T, dtype=np.float64)
        if T > 1:
            right[:-1] = mu_edges
            left[1:] = mu_edges

        a = np.empty((T, Hf.size), dtype=np.float64)
        for t in range(T):
            a[t] = absH2 + lam + left[t] + right[t]

        l = np.zeros((T, Hf.size), dtype=np.float64)
        u = np.zeros((T, Hf.size), dtype=np.float64)
        if T > 1:
            l[1:] = -mu_edges[:, None]
            u[:-1] = -mu_edges[:, None]

        X = thomas_batch(l.astype(np.complex128), a.astype(np.complex128), u.astype(np.complex128), d)

        rho_seq = np.zeros((T, N, N, N), dtype=np.float32)
        for t in range(T):
            Xp = X[t].reshape(N_pad, N_pad, N_pad)
            x_p = ifftn(Xp)
            x = self.forward._crop_center(x_p, N)
            rho_seq[t] = np.abs(x).astype(np.float32)

        if self.normalize == "per_frame":
            for t in range(T):
                mn, mx = float(rho_seq[t].min()), float(rho_seq[t].max())
                rho_seq[t] = (rho_seq[t] - mn) / (mx - mn + 1e-8)
        else:
            mn, mx = float(rho_seq.min()), float(rho_seq.max())
            rho_seq = (rho_seq - mn) / (mx - mn + 1e-8)
        return rho_seq


def warp_scores(rho_seq):
    T = rho_seq.shape[0]
    w = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        w[t] = float(np.mean(np.abs(rho_seq[t] - rho_seq[t-1])))
    return w


def event_candidate_score(e: EventRec, warp: np.ndarray, T: int):
    t = time_bin(e.local_time, T)
    w = warp[t]
    if t < T-1:
        w = max(w, warp[t+1])
    return float(w * contradiction_score(e)), t, w


def plot_overview(resource, contr, mu_edges, warp, out_png: str):
    T = len(resource)
    x = np.arange(T)
    fig = plt.figure(figsize=(10.5, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, resource, marker="o", label="Resource C(t)")
    ax.plot(x, contr, marker="o", label="Contradiction K(t)")
    ax.plot(x, warp, marker="o", label="Warp W(t)")
    xe = np.arange(T-1) + 0.5
    ax.plot(xe, mu_edges, marker="s", linestyle="--", label="mu_edges")
    ax.set_xticks(x)
    ax.set_xlabel("time frame t (0..T-1)")
    ax.set_ylim(0, max(0.25, float(max(resource.max(), contr.max(), warp.max(), mu_edges.max()) * 1.15)))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)


def run(json_path: str, kimura2_txt: str, out_dir: str, T=9, N=48, pad=2, noise=0.02, reg_lambda=1e-3,
        mu_min=0.01, mu_max=0.12, power=1.0, alpha=0.6, mu_floor=0.006, topk=10):
    events = [score_axes(e) for e in load_events(json_path)]

    resource_t, contr_t = frame_series(events, T=T)
    mu_edges, _, _ = make_mu_edges(resource_t, contr_t, mu_min=mu_min, mu_max=mu_max,
                                   power=power, alpha=alpha, mu_floor=mu_floor)

    rho_truth_seq, counts = build_rho_sequence(events, T=T, N=N)

    kim2 = load_kimura2_from_txt(kimura2_txt)
    forward = kim2.PhysicsForwardModel(size=N)

    G_seq = np.zeros((T, N, N, N), dtype=np.complex128)
    for t in range(T):
        if rho_truth_seq[t].max() <= 0:
            continue
        G_seq[t] = forward.simulate_scattering(rho_truth_seq[t], noise_level=noise, pad_factor=pad, seed=0)

    solver = KimuraSolver4DDynamic(forward, size=N, reg_lambda=reg_lambda, pad_factor=pad, normalize="global")
    rho_rec_seq = solver.reconstruct_sequence(G_seq, mu_edges=mu_edges)
    warp = warp_scores(rho_rec_seq)

    idx = np.argsort(warp)[::-1]
    print("\n=== Top warp frames ===")
    for r, t in enumerate(idx[:5], 1):
        print(f"{r:02d} | t={t} | warp={warp[t]:.6f} | events={counts[t]} | C={resource_t[t]:.2f} | K={contr_t[t]:.2f}")

    scored = []
    for e in events:
        s, t, w = event_candidate_score(e, warp, T)
        scored.append((s, t, w, e))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\n=== Turning-point candidate events TOP {topk} (warp * contradiction) ===")
    for i, (s, t, w, e) in enumerate(scored[:topk], 1):
        print(f"{i:02d} | score={s:.6f} | t={t} | warp~={w:.6f}")
        print(f"    {e.chapter_title} | step={e.global_step} | local_time={e.local_time:.2f}")
        print(f"    A={e.A_approach:.2f} B={e.B_threat:.2f} C={e.C_resource:.2f} | {e.event}")
        if e.desc:
            print(f"    desc: {e.desc}")

    import os
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "turning_points.png")
    plot_overview(resource_t, contr_t, mu_edges, warp, out_png)
    print(f"\nSaved: {out_png}")
    return out_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="path to event json")
    p.add_argument("--kimura2", default="src/kimura2.txt", help="path to kimura2.txt")
    p.add_argument("--out_dir", default="assets", help="output directory for figures")
    p.add_argument("--T", type=int, default=9)
    p.add_argument("--N", type=int, default=48)
    args = p.parse_args()
    run(args.json, args.kimura2, args.out_dir, T=args.T, N=args.N)

if __name__ == "__main__":
    main()

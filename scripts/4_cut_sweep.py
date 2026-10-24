"""3주차 보충: cut 위치 실험 — 모델이 동작을 보는가, 파일별 잡음을 보는가

[4단계] 결과물: outputs/week_3/cut_sweep.csv · figures/week_3/cut_sweep.png
       1_preprocessing.py 의 cut 시작점을 0~3 s 로 옮겨 가며 특징을 다시 뽑고, RF 로 trial-wise 와 LOSO 정확도를 잰다.
       표본은 2_training.py 와 같은 시행 단위(475차원).
       dataset.npz 는 건드리지 않는다 (특징은 메모리에서만 만든다).

실행: python scripts/4_cut_sweep.py [--starts 0 0.5 1 1.5 2 2.5 3]

보려는 것: 동작 구간(1~3 s)보다 휴식 구간(0~2 s, 3~5 s)에서 trial-wise 정확도가 높다면, 모델이 근육 활동이 아니라
파일(피험자 x 클래스 CSV = 세션)마다 다른 잡음 수준을 외우는 것이다. 그 서명은 다른 사람에게 옮겨가지 않으므로
LOSO 는 어느 구간에서도 오르지 않아야 한다. 흐름도: mermaid/cut_sweep.md
"""
import argparse
import importlib
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from data import ROOT, FS, load_trials
from loso import FOLDS, to_trials

pp = importlib.import_module("1_preprocessing")        # 숫자로 시작하는 모듈은 import 문으로 못 불러온다
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "outputs" / "week_3" / "cut_sweep.csv"
FIG = ROOT / "figures" / "week_3" / "cut_sweep.png"
SEED = 42
RF = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)   # 2_training.py 의 RF 와 같은 설정
CHANCE = 1 / 8


def features_at(Xf: np.ndarray, trials: pd.DataFrame, start_s: float, le: LabelEncoder) -> dict:
    """cut 시작점만 바꿔 1_preprocessing.py 와 같은 창·특징을 만들고 시행 단위(475차원)로 묶는다 (필터는 이미 적용된 Xf)"""
    start = int(start_s * FS)
    feats = np.stack([np.stack([pp.td_features(w) for w in pp.make_windows(pp.cut(x, start))]) for x in Xf])
    n, k, d = feats.shape
    rep = lambda c: np.repeat(trials[c].astype(str).to_numpy(), k)
    return to_trials({"X": feats.reshape(n * k, d), "y": le.transform(rep("label")), "subject": rep("subject"),
                      "trial": np.repeat(np.arange(n), k)})


def rf_cv(d: dict, split: str) -> tuple[float, float]:
    """RF 로 split 교차검증, fold 정확도의 평균·표준편차"""
    acc = [clone(RF).fit(Xtr, ytr).score(Xte, yte) for _, Xtr, ytr, Xte, yte, _ in FOLDS[split](d)]
    return float(np.mean(acc)), float(np.std(acc))


def plot(res: pd.DataFrame, env_t: np.ndarray, env: np.ndarray, path) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=False, gridspec_kw={"height_ratios": [1, 2]})
    # 위: 시행 평균 RMS 포락선. 동작이 어디 있는지
    a1.plot(env_t, env * 1e3, color="k", lw=1.2)
    a1.set_ylabel("mean RMS (x1e-3)"); a1.set_xlabel("time in trial (s)"); a1.set_xlim(0, 5); a1.grid(alpha=0.3)
    a1.set_title("where the motion is: trial-averaged RMS envelope (all trials, 5 channels)")
    for _, r in res.iterrows():
        a1.axvspan(r.start_s, r.start_s + 2, color="#3fb9c4", alpha=0.08)
    # 아래: cut 시작점별 정확도
    a2.errorbar(res.start_s, res.trialwise_acc, yerr=res.trialwise_std, marker="o", color="#1f7a85", lw=2, capsize=3, label="trial-wise 5-fold (same subjects in train)")
    a2.errorbar(res.start_s, res.loso_acc, yerr=res.loso_std, marker="s", color="#c4603f", lw=2, capsize=3, label="LOSO 8-fold (unseen subject)")
    a2.axhline(CHANCE, color="gray", ls=":", lw=1, label="chance (1/8)")
    a2.axvline(1.0, color="k", ls="--", lw=0.8, label="default cut 1.0–3.0 s (2_training.py)")
    a2.set_xlabel("cut start (s); each cut is 2 s long"); a2.set_ylabel("RF accuracy"); a2.set_ylim(0, 1); a2.grid(alpha=0.3)
    a2.set_xticks(res.start_s, [f"{s:.1f}–{s + 2:.1f}" for s in res.start_s], rotation=30)
    a2.legend(loc="center right", fontsize=8)
    a2.set_title("RF accuracy vs. which 2 s of the trial is used")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="cut 시작점을 바꿔 가며 RF trial-wise · LOSO 정확도를 잰다")
    parser.add_argument("--starts", type=float, nargs="+", default=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], help="cut 시작점 (초). 구간 길이는 2 s 고정")
    args = parser.parse_args(argv)

    X_all, trials = load_trials()
    Xf = np.stack([pp.preprocess(x) for x in X_all])
    le = LabelEncoder().fit(trials["label"].astype(str))
    # 시행 평균 RMS 포락선 (100 ms 창, 채널 평균). 동작 위치 표시용
    win = int(0.1 * FS); n_env = X_all.shape[1] // win
    env = np.sqrt((Xf[:, :n_env * win].reshape(len(Xf), n_env, win, -1) ** 2).mean(2)).mean((0, 2))
    env_t = (np.arange(n_env) + 0.5) * win / FS

    rows = []
    for s in args.starts:
        t0 = time.time()
        d = features_at(Xf, trials, s, le)
        seg_rms = float(np.sqrt((Xf[:, int(s * FS):int(s * FS) + 2000] ** 2).mean()))
        row = {"start_s": s, "end_s": s + 2, "segment_rms": seg_rms}
        for split in ("trialwise", "loso"):
            row[f"{split}_acc"], row[f"{split}_std"] = rf_cv(d, split)
        rows.append(row)
        print(f"cut {s:.1f}~{s + 2:.1f} s  구간 RMS {seg_rms * 1e3:5.2f}e-3  "
              f"trial-wise {row['trialwise_acc']:.3f} ± {row['trialwise_std']:.3f}  LOSO {row['loso_acc']:.3f} ± {row['loso_std']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True); res.to_csv(OUT, index=False)
    plot(res, env_t, env, FIG)
    pd.set_option("display.width", 200)
    print("\n" + res.set_index("start_s")[[c for c in res.columns if c.endswith("_acc")]].round(3).to_string())
    print(f"우연 {CHANCE:.3f}")
    print(f"저장: {OUT}, {FIG}")


if __name__ == "__main__":
    main()

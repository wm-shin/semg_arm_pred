"""3주차 보충: RF 검증 히스토리 (트리 수 · 트리 깊이에 따른 검증 정확도)

[3단계] 결과물: outputs/week_3/{split}/rf_val_trees.csv · rf_val_depth.csv, figures/week_3/rf_val_{split}.png
       2_training.py 의 RF_TREES · RF_MAX_DEPTH 를 정하는 근거. 테스트 fold 는 건드리지 않는다.

실행: python scripts/3_rf_validation.py [--data outputs/dataset.npz] [--split loso|trialwise]

딥러닝의 epoch별 검증 곡선에 해당하는 것을 RF로 만든다.
  트리 수 곡선   트리 N_MAX개를 한 번 학습해 두고, 앞에서 k개만 다수결했을 때의 검증 정확도를 k=1..N_MAX 로 그린다
  깊이 곡선     max_depth 를 바꿔 가며 학습해 검증 정확도를 그린다
검증 데이터는 외부 fold 의 학습 쪽 안에서 뗀다 (loso: 학습 7명 중 1명, trialwise: 학습 시행의 20%를 피험자x라벨로 층화).
외부 fold 마다 곡선을 하나씩 만들고 평균 ± 표준편차로 모은다 (2_training.py 의 fold 평균과 같은 방식).
흐름도: mermaid/rf_validation.md
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data import ROOT
from loso import DATASET, FOLDS, load_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "outputs" / "week_3"
FIG = ROOT / "figures" / "week_3"
SEED = 42
N_MAX = 300                                                  # 트리 수 곡선의 최대 트리 수
DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20, 30, None]              # 깊이 곡선. None = 제한 없음 (2_training.py 기본)
DEPTH_TREES = 100                                            # 깊이 곡선을 그릴 때의 트리 수 (2_training.py 기본)


def inner_split(d: dict, tr: np.ndarray, split: str, test_name: str) -> tuple[np.ndarray, np.ndarray]:
    """외부 fold 의 학습 행 tr 을 내부 학습/검증으로 나눈다. 테스트 행은 들어오지 않는다"""
    if split == "loso":
        subs = np.unique(d["subject"][tr])                    # 학습 피험자 7명
        val_sub = subs[(np.searchsorted(subs, test_name)) % len(subs)]   # 테스트 피험자 다음 사람 (한 바퀴 돌면 8명이 한 번씩 검증)
        is_val = d["subject"][tr] == val_sub
    else:
        trial = d["trial"][tr]
        trials, first = np.unique(trial, return_index=True)
        strata = np.char.add(d["subject"][tr][first].astype(str), d["y"][tr][first].astype(str))   # 피험자 x 라벨 (loso.py 와 같은 기준)
        _, val_t = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(trials, strata))
        is_val = np.isin(trial, trials[val_t])
    return tr[~is_val], tr[is_val]


def trees_curve(Xa, ya, Xv, yv) -> np.ndarray:
    """트리 N_MAX개 학습 → 앞 k개 다수결의 검증 정확도 (k=1..N_MAX). 확률을 누적 평균해 argmax"""
    rf = RandomForestClassifier(n_estimators=N_MAX, random_state=SEED, n_jobs=-1).fit(Xa, ya)
    proba = np.stack([t.predict_proba(Xv) for t in rf.estimators_])        # (N_MAX, 검증 창, 클래스)
    cum = np.cumsum(proba, axis=0) / np.arange(1, N_MAX + 1)[:, None, None]
    return (rf.classes_[cum.argmax(2)] == yv).mean(1)                      # (N_MAX,)


def depth_curve(Xa, ya, Xv, yv) -> np.ndarray:
    """max_depth 마다 학습해 검증 정확도"""
    return np.array([RandomForestClassifier(n_estimators=DEPTH_TREES, max_depth=md, random_state=SEED, n_jobs=-1)
                     .fit(Xa, ya).score(Xv, yv) for md in DEPTHS])


def summarize(curves: dict[str, np.ndarray], index, name: str) -> pd.DataFrame:
    """fold별 곡선을 열로 두고 mean · std 열을 붙인다"""
    df = pd.DataFrame(curves, index=pd.Index(index, name=name))
    df["mean"], df["std"] = df.mean(axis=1), df.std(axis=1)
    return df


def plot(trees: pd.DataFrame, depth: pd.DataFrame, split: str, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    folds = [c for c in trees.columns if c not in ("mean", "std")]
    for ax, df, xlabel in ((a1, trees, "number of trees"), (a2, depth, "max_depth (R = unlimited)")):
        x = np.arange(len(df)) if df is depth else df.index
        for c in folds:
            ax.plot(x, df[c], color="gray", lw=0.6, alpha=0.5)
        ax.fill_between(x, df["mean"] - df["std"], df["mean"] + df["std"], color="#3fb9c4", alpha=0.25, label="mean ± std")
        ax.plot(x, df["mean"], color="#1f7a85", lw=2, label=f"mean over {len(folds)} folds")
        ax.set_xlabel(xlabel); ax.set_ylabel("validation accuracy"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    a1.axvline(DEPTH_TREES, color="k", ls="--", lw=0.8); a1.set_title(f"RF: trees  (dashed = {DEPTH_TREES}, 2_training.py default)")
    a2.set_xticks(range(len(DEPTHS)), ["R" if d is None else str(d) for d in DEPTHS])
    a2.set_title(f"RF: max_depth  ({DEPTH_TREES} trees)")
    fig.suptitle(f"RF validation history — {split} (validation held out inside each training fold)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RF 검증 히스토리 (트리 수 · 깊이)")
    parser.add_argument("--data", default=DATASET)
    parser.add_argument("--split", choices=FOLDS, default="loso")
    args = parser.parse_args(argv)
    out = OUT / args.split
    out.mkdir(parents=True, exist_ok=True)

    d, le = load_dataset(args.data)
    n = len(d["X"])
    print(f"X {d['X'].shape} (시행 단위) · 분할 {args.split}")
    tcurves, dcurves = {}, {}
    for name, _, _, _, _, te in FOLDS[args.split](d):
        tr = np.setdiff1d(np.arange(n), te)                   # 외부 학습 행 (테스트 제외)
        a, v = inner_split(d, tr, args.split, name)
        sc = StandardScaler().fit(d["X"][a])                   # 내부 학습 통계로만 표준화
        Xa, Xv = sc.transform(d["X"][a]), sc.transform(d["X"][v])
        ya, yv = d["y"][a], d["y"][v]
        tcurves[name] = trees_curve(Xa, ya, Xv, yv)
        dcurves[name] = depth_curve(Xa, ya, Xv, yv)
        vdesc = np.unique(d["subject"][v])[0] if args.split == "loso" else f"{len(np.unique(d['trial'][v]))}시행"
        print(f"{name}: 내부 학습 {len(a)} / 검증 {len(v)} 표본 ({vdesc})  "
              f"트리 100개 {tcurves[name][99]:.3f} · 300개 {tcurves[name][-1]:.3f} · "
              f"깊이 최적 {['R' if x is None else x for x in DEPTHS][dcurves[name].argmax()]} ({dcurves[name].max():.3f})", flush=True)

    trees = summarize(tcurves, np.arange(1, N_MAX + 1), "n_trees")
    depth = summarize(dcurves, ["R" if x is None else x for x in DEPTHS], "max_depth")
    trees.to_csv(out / "rf_val_trees.csv"); depth.to_csv(out / "rf_val_depth.csv")
    plot(trees, depth, args.split, FIG / f"rf_val_{args.split}.png")

    pd.set_option("display.width", 160)
    print(f"\n트리 수에 따른 검증 정확도 (fold 평균 ± 표준편차)")
    print(trees.loc[[1, 5, 10, 20, 50, 100, 150, 200, 300], ["mean", "std"]].round(4).T.to_string())
    print(f"\nmax_depth 에 따른 검증 정확도 (트리 {DEPTH_TREES}개)")
    print(depth[["mean", "std"]].round(4).T.to_string())
    k95 = int(np.argmax(trees["mean"] >= trees["mean"].max() - 0.005)) + 1
    print(f"\n요약: 트리 수는 {k95}개부터 최고치 -0.5%p 안 (최고 {trees['mean'].max():.4f} @ {trees['mean'].idxmax()}개) · "
          f"깊이 최적 {depth['mean'].idxmax()} ({depth['mean'].max():.4f}, 제한 없음 R = {depth.loc['R', 'mean']:.4f})")
    print(f"저장: {out}/rf_val_trees.csv · rf_val_depth.csv, {FIG}/rf_val_{args.split}.png")


if __name__ == "__main__":
    main()

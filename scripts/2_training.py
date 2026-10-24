"""3주차 실습: 교차검증으로 SVM · RF · XGBoost 학습·평가·검정

[2단계] 결과물: outputs/week_3/{split}/*.csv · figures/week_3/confusion8_{split}.png. --split loso 와 trialwise 로 두 번 실행한다.

실행: python scripts/2_training.py [--data outputs/dataset.npz] [--split loso|trialwise] [--out outputs/week_3]

dataset.npz를 읽어 --split이 고른 분할 생성기(scripts/loso.py의 FOLDS)로 fold를 돌리면서 세 모델을
같은 fold·같은 표준화로 학습한다. fold마다 지표 4개와 시간을 rows에, 창마다 (실제, 예측)을 preds에 모아
제출물 ② 성능 비교표 · ③ fold별 정확도 · ④ 대응표본 t-검정 · ⑤ 혼동행렬 · ⑥ 계산 비용을 만든다.
기본은 LOSO 8-fold(처음 보는 사람), --split trialwise 는 논문·강의 방식의 시행 단위 5-fold.
표본은 시행 하나 (창 19개 이어 붙인 475차원, 1920개) = 논문 Table 1 의 구성. load_dataset 이 만든다.
분할·표준화는 scripts/loso.py, 특징 추출은 scripts/1_preprocessing.py. 흐름도: mermaid/training.md
"""
import argparse
import time
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder  # 타입 표기용
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data import ROOT
from loso import DATASET, FOLDS, load_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "outputs" / "week_3"
FIG = ROOT / "figures" / "week_3"
SEED = 42                      # 세 모델 공통 random_state. 고정해야 다음 실행에서도 같은 결과

# ---- 하이퍼파라미터 (논문 2.4절 값. SVM·RF는 sklearn 기본값, XGBoost만 예비 튜닝값) ------------
# 여기 값을 바꾸고 다시 실행하면 된다. 바꾼 실행은 --out 으로 다른 폴더에 저장해 기준 결과를 덮어쓰지 않게 할 것.
SVM_C = 1.0                    # 오분류 벌점. 크면 경계가 학습 데이터에 더 붙는다 (과적합 쪽)
SVM_GAMMA = "scale"            # RBF 커널 폭. 'scale' = 1 / (특징 수 x 분산). 숫자로 주면 클수록 경계가 구불구불
RF_TREES = 100                 # 트리 개수. 늘리면 분산이 줄지만 100 이상은 효과가 작다
RF_MAX_DEPTH = None            # 트리 깊이 제한. None = 잎이 순수해질 때까지. 과적합이면 10~20으로 제한
XGB_TREES = 400                # 부스팅 라운드 수. learning_rate 와 반비례로 맞춘다
XGB_MAX_DEPTH = 6              # 트리 하나의 깊이
XGB_LR = 0.05                  # 한 라운드가 고치는 크기. 작을수록 보수적, 라운드 수가 더 필요
XGB_SUBSAMPLE = 0.9            # 라운드마다 쓰는 행 비율
XGB_COLSAMPLE = 0.8            # 트리마다 쓰는 특징 비율
XGB_LAMBDA = 1.0               # L2 규제

HPARAMS = {"svm_C": SVM_C, "svm_gamma": SVM_GAMMA, "rf_trees": RF_TREES, "rf_depth": RF_MAX_DEPTH,
           "xgb_trees": XGB_TREES, "xgb_depth": XGB_MAX_DEPTH, "xgb_lr": XGB_LR, "xgb_sub": XGB_SUBSAMPLE,
           "xgb_col": XGB_COLSAMPLE, "xgb_lambda": XGB_LAMBDA}

MODELS = {
    "SVM": SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, decision_function_shape="ovr"),
    "RF": RandomForestClassifier(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH, random_state=SEED, n_jobs=-1),
    "XGB": XGBClassifier(n_estimators=XGB_TREES, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LR,
                         subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE, reg_lambda=XGB_LAMBDA,
                         tree_method="hist", random_state=SEED),
}
METRICS = ["acc", "precision", "recall", "f1"]


def run_folds(d: dict, folds, le: LabelEncoder) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fold x 모델마다 clone → fit → predict. (scores: fold x 모델 1행, preds: 표본마다 1행)"""
    rows, preds = [], []
    for name, Xtr, ytr, Xte, yte, te in folds(d):
        for k, proto in MODELS.items():
            m = clone(proto)                                   # fold마다 새 모델
            t0 = time.perf_counter(); m.fit(Xtr, ytr); train_s = time.perf_counter() - t0
            t0 = time.perf_counter(); p = m.predict(Xte); infer_ms = (time.perf_counter() - t0) / len(Xte) * 1e3
            rows.append({"fold": name, "model": k,
                         "acc": accuracy_score(yte, p),
                         "precision": precision_score(yte, p, average="macro", zero_division=0),
                         "recall": recall_score(yte, p, average="macro"),
                         "f1": f1_score(yte, p, average="macro"),
                         "train_s": train_s, "infer_ms": infer_ms})
            preds.append(pd.DataFrame({"fold": name, "subject": d["subject"][te], "model": k,
                                       "y": le.inverse_transform(yte), "pred": le.inverse_transform(p)}))
            print(f"{name} {k:3s}: acc {rows[-1]['acc']:.3f} · f1 {rows[-1]['f1']:.3f} · "
                  f"학습 {train_s:.1f}s · 추론 {infer_ms:.3f} ms/표본", flush=True)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def mean_std(scores: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """모델별 '평균 ± 표준편차' 문자열 표 (② 성능 비교표, ⑥ 계산 비용)"""
    g = scores.groupby("model", sort=False)[cols]
    return g.mean().combine(g.std(), lambda m, s: m.map("{:.4f}".format) + " ± " + s.map("{:.4f}".format))


def fold_table(scores: pd.DataFrame) -> pd.DataFrame:
    """③ fold별 정확도 피벗 + 평균·표준편차 행"""
    t = scores.pivot(index="fold", columns="model", values="acc")[list(MODELS)]
    t.loc["mean"], t.loc["std"] = t.mean(), t.std()
    return t


def paired_ttest(scores: pd.DataFrame) -> pd.DataFrame:
    """④ fold별 정확도를 같은 fold끼리 짝지어 세 쌍 대응표본 t-검정"""
    acc = scores.pivot(index="fold", columns="model", values="acc")
    rows = []
    for a, b in combinations(MODELS, 2):
        t, p = ttest_rel(acc[a], acc[b])
        verdict = "검정 불가" if not np.isfinite(p) else ("유의함" if p < 0.05 else "유의하지 않음")
        rows.append({"pair": f"{a} vs {b}", "mean_diff": acc[a].mean() - acc[b].mean(), "t": t, "p": p, "verdict": verdict})
    return pd.DataFrame(rows)


def plot_confusion(preds: pd.DataFrame, classes: np.ndarray, path: Path) -> None:
    """⑤ 모델별 8x8 혼동행렬 (전체 fold 예측 합산, 행 기준 %) 1행 3열"""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 5))
    for ax, k in zip(axes, MODELS):
        q = preds[preds.model == k]
        cm = confusion_matrix(q.y, q.pred, labels=classes)
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] > 50 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_title(f"{k}  acc {np.trace(cm) / cm.sum():.3f}")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle(f"{path.stem}  (row %, all folds pooled)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="교차검증으로 세 모델을 학습·평가·검정한다")
    parser.add_argument("--data", default=DATASET)
    parser.add_argument("--split", choices=FOLDS, default="loso")
    parser.add_argument("--out", default=OUT, type=Path)
    args = parser.parse_args(argv)
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)

    d, le = load_dataset(args.data)                               # 라벨 0~7 정수화 + 시행 단위 (scripts/loso.py)
    print(f"X {d['X'].shape} (시행 단위) · 클래스 {len(le.classes_)} · 분할 {args.split}")
    print("설정:", " ".join(f"{k}={v}" for k, v in HPARAMS.items()))

    scores, preds = run_folds(d, FOLDS[args.split], le)
    scores.to_csv(out / "scores.csv", index=False)
    preds.to_csv(out / "preds.csv", index=False)
    tt = paired_ttest(scores)
    tt.to_csv(out / "ttest.csv", index=False)
    plot_confusion(preds, le.classes_, FIG / f"confusion8_{args.split}.png")

    pd.set_option("display.width", 160)
    print(f"\n② 성능 비교표 (fold 평균 ± 표준편차, macro)\n{mean_std(scores, METRICS)}")
    print(f"\n③ fold별 정확도\n{fold_table(scores).round(4)}")
    print(f"\n④ 대응표본 t-검정 (fold별 정확도, 자유도 {scores.fold.nunique() - 1})\n{tt.round(4).to_string(index=False)}")
    print(f"\n⑥ 계산 비용 (학습 s · 추론 ms/표본, fold 평균 ± 표준편차)\n{mean_std(scores, ['train_s', 'infer_ms'])}")
    print(f"\n저장: {out}/scores.csv · preds.csv · ttest.csv, {FIG}/confusion8_{args.split}.png")


if __name__ == "__main__":
    main()

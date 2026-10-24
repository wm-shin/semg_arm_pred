"""분할 생성기 (LOSO · trial-wise) + fold 안 표준화

[불러오기용 모듈] 번호 붙은 스크립트가 import해서 쓴다. 직접 실행하면 확인 출력만 한다 (결과물 없음).

실행: python scripts/loso.py [--data outputs/dataset.npz] [--split loso|trialwise]  → fold별 모양 확인
재사용: from loso import FOLDS, load_dataset                   (scripts/2_training.py 가 이렇게 쓴다)

dataset.npz (scripts/1_preprocessing.py가 만든 표준화 전 창 단위 특징)를 시행 단위로 묶은 뒤 두 가지 방식으로 나눈다.
  loso      피험자 한 명씩 테스트 (8명 → 8-fold). 처음 보는 사람에 대한 성능
  trialwise 시행 1920개를 피험자x라벨로 층화해 5-fold (논문·강의 방식). 같은 사람의 다른 시행에 대한 성능
어느 쪽이든 한 시행의 창 19개는 같은 쪽으로 가고, 표준화는 fold마다 학습 쪽 통계로만 한다.
표본 = 시행 하나 (load_dataset 이 창 19개를 이어 붙여 475차원 x 1920개로 만든다. 논문 Table 1의 구성).
흐름도: mermaid/loso.md
"""
import argparse
from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data import ROOT

DATASET = ROOT / "outputs" / "dataset.npz"

def to_trials(d: dict) -> dict:
    """1_preprocessing.py 가 저장한 창 단위 배열 (36480행) → 시행 단위 (1920행). 한 시행의 창 19개를 순서대로 이어 붙인다.

    dataset.npz 의 행 순서가 시행 x 창(19)이라는 전제를 확인한 뒤 reshape 한다.
    """
    X, trial = d["X"], d["trial"]
    n = len(np.unique(trial)); k = len(X) // n
    if not np.array_equal(trial, np.repeat(np.arange(n), k)):
        raise ValueError("행 순서가 시행 x 창 이 아니라서 시행 단위로 묶을 수 없다")
    out = {"X": X.reshape(n, k * X.shape[1]), "y": d["y"][::k], "subject": d["subject"][::k], "trial": trial[::k]}
    if "feature_names" in d:
        out["feature_names"] = np.array([f"w{w + 1:02d}_{f}" for w in range(k) for f in d["feature_names"]])
    return out


def load_dataset(path=DATASET) -> tuple[dict, LabelEncoder]:
    """dataset.npz 를 읽어 라벨을 0~7 정수로 바꾸고 (XGBoost용) 시행 단위 dict 와 LabelEncoder 를 돌려준다.

    표본 = 시행 하나 (창 19개 x 25차원 = 475차원, 1920개). 논문 Table 1 (학습 1536 + 테스트 384) 과 같은 구성.
    LabelEncoder 는 라벨 이름만 보므로 전체 y 로 fit 해도 테스트 정보가 새지 않는다.
    """
    raw = np.load(path); le = LabelEncoder()
    d = {"X": raw["X"], "y": le.fit_transform(raw["y"]), "subject": raw["subject"], "trial": raw["trial"],
         "feature_names": raw["feature_names"]}
    return to_trials(d), le

Fold = tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""(fold 이름, Xtr, ytr, Xte, yte, te). te는 테스트 창의 행 번호 → d["subject"][te] 등으로 원본 열을 찾는다"""


def _fold(d, name: str, tr: np.ndarray, te: np.ndarray) -> Fold:
    """행 번호 tr/te로 나누고 학습 쪽 통계로만 표준화한다 (테스트 정보가 새지 않게)"""
    X, y = d["X"], d["y"]
    sc = StandardScaler().fit(X[tr])
    return name, sc.transform(X[tr]), y[tr], sc.transform(X[te]), y[te], te


def loso_folds(d) -> Iterator[Fold]:
    """피험자 한 명씩 테스트로 돌린다. fold 이름 = 테스트 피험자 'a'~'h'

    d: np.load(dataset.npz) 또는 같은 키(X, y, subject)를 가진 dict
    """
    subject = d["subject"]
    for tr, te in LeaveOneGroupOut().split(d["X"], d["y"], groups=subject):
        yield _fold(d, subject[te][0], tr, te)


def trialwise_folds(d, k: int = 5, seed: int = 42) -> Iterator[Fold]:
    """시행 단위 층화 k-fold (논문·강의 방식). fold 이름 = 'fold1'~'fold5'

    시행 1920개를 (피험자 x 라벨) 64칸의 비율을 지키며 k묶음으로 나눈다. 칸마다 시행 30개라
    fold마다 정확히 6개씩 테스트에 들어가고, 피험자 비율과 클래스 비율이 모두 fold 간에 같다.
    (라벨로만 층화하면 fold마다 피험자별 시행 수가 36~60개로 흔들리고 어떤 피험자x클래스 칸은 0개가 된다.)
    테스트 fold에 8명이 다 섞여 있으므로 학습 때 본 사람의 다른 시행을 맞히는 과제다.
    d: 같은 키(X, y, subject, trial)를 가진 dict
    """
    trial = d["trial"]
    trials, first = np.unique(trial, return_index=True)   # 시행 번호, 그 시행의 첫 행
    strata = np.char.add(d["subject"][first].astype(str), d["y"][first].astype(str))   # 'a0', 'a1', … 64종
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for i, (tr_t, te_t) in enumerate(skf.split(trials, strata), 1):
        tr = np.flatnonzero(np.isin(trial, trials[tr_t]))
        te = np.flatnonzero(np.isin(trial, trials[te_t]))
        yield _fold(d, f"fold{i}", tr, te)


FOLDS = {"loso": loso_folds, "trialwise": trialwise_folds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="분할을 확인한다")
    parser.add_argument("--data", default=DATASET)
    parser.add_argument("--split", choices=FOLDS, default="loso")
    args = parser.parse_args(argv)

    d, _ = load_dataset(args.data)
    print(f"X {d['X'].shape} (시행 단위) · 피험자 {len(np.unique(d['subject']))}명 → {args.split}")
    for name, Xtr, ytr, Xte, yte, te in FOLDS[args.split](d):
        subj = "".join(np.unique(d["subject"][te]))
        print(f"{name}: 학습 {Xtr.shape[0]} / 테스트 {Xte.shape[0]} 표본 (피험자 {subj})  "
              f"학습 평균 {np.abs(Xtr.mean(0)).max():.0e} · 테스트 평균 최대 {np.abs(Xte.mean(0)).max():.2f}")


if __name__ == "__main__":
    main()

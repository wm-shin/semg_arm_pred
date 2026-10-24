"""분할 생성기 (LOSO · trial-wise) + fold 안 표준화

[불러오기용 모듈] 번호 붙은 스크립트가 import해서 쓴다. 직접 실행하면 확인 출력만 한다 (결과물 없음).

실행: python scripts/loso.py [--data outputs/dataset.npz] [--split loso|trialwise]  → fold별 모양 확인
재사용: from loso import FOLDS                                 (scripts/2_training.py 가 이렇게 쓴다)

dataset.npz (scripts/1_preprocessing.py가 만든 표준화 전 특징 전체)를 받아 두 가지 방식으로 나눈다.
  loso      피험자 한 명씩 테스트 (8명 → 8-fold). 처음 보는 사람에 대한 성능
  trialwise 시행 1920개를 라벨로 층화해 5-fold (논문·강의 방식). 같은 사람의 다른 시행에 대한 성능
어느 쪽이든 한 시행의 창 19개는 같은 쪽으로 가고, 표준화는 fold마다 학습 쪽 통계로만 한다.
흐름도: docs/loso.md
"""
import argparse
from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data import ROOT

DATASET = ROOT / "outputs" / "dataset.npz"

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

    시행 1920개를 라벨 비율을 지키며 k묶음으로 나눈 뒤 창 36480개로 펼친다.
    테스트 fold에 8명이 다 섞여 있으므로 학습 때 본 사람의 다른 시행을 맞히는 과제다.
    d: 같은 키(X, y, trial)를 가진 dict
    """
    trial = d["trial"]
    trials, first = np.unique(trial, return_index=True)   # 시행 번호, 그 시행의 첫 창 행
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for i, (tr_t, te_t) in enumerate(skf.split(trials, d["y"][first]), 1):
        tr = np.flatnonzero(np.isin(trial, trials[tr_t]))
        te = np.flatnonzero(np.isin(trial, trials[te_t]))
        yield _fold(d, f"fold{i}", tr, te)


FOLDS = {"loso": loso_folds, "trialwise": trialwise_folds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="분할을 확인한다")
    parser.add_argument("--data", default=DATASET)
    parser.add_argument("--split", choices=FOLDS, default="loso")
    args = parser.parse_args(argv)

    d = np.load(args.data)
    print(f"X {d['X'].shape} · 피험자 {len(np.unique(d['subject']))}명 · 시행 {len(np.unique(d['trial']))}개 → {args.split}")
    for name, Xtr, ytr, Xte, yte, te in FOLDS[args.split](d):
        subj = "".join(np.unique(d["subject"][te]))
        print(f"{name}: 학습 {Xtr.shape[0]}창 / 테스트 {Xte.shape[0]}창 (피험자 {subj}, 시행 {len(np.unique(d['trial'][te]))}개)  "
              f"학습 평균 {np.abs(Xtr.mean(0)).max():.0e} · 테스트 평균 최대 {np.abs(Xte.mean(0)).max():.2f}")


if __name__ == "__main__":
    main()

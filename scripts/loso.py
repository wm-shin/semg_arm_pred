"""LOSO (Leave-One-Subject-Out) 분할 + fold 안 표준화

실행: python scripts/loso.py [--data outputs/dataset.npz]   → fold별 학습/테스트 모양 확인
재사용: from loso import loso_folds                           (scripts/training.py 가 이렇게 쓴다)

dataset.npz (scripts/preprocessing.py가 만든 표준화 전 특징 전체)를 받아 피험자 한 명씩
테스트로 돌린다 (8명 → 8-fold). 표준화는 fold마다 학습 피험자 7명의 통계로만 한다.
흐름도: docs/loso.md
"""
import argparse
from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from data import ROOT

DATASET = ROOT / "outputs" / "dataset.npz"


def loso_folds(d) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """피험자 한 명씩 테스트로 돌리며 (테스트 피험자, Xtr, ytr, Xte, yte)를 낸다.

    표준화 통계는 매 fold의 학습 피험자에서만 구한다 (테스트 피험자 정보가 새지 않게).
    d: np.load(dataset.npz) 또는 같은 키(X, y, subject)를 가진 dict
    """
    X, y, subject = d["X"], d["y"], d["subject"]
    for tr, te in LeaveOneGroupOut().split(X, y, groups=subject):
        sc = StandardScaler().fit(X[tr])
        yield subject[te][0], sc.transform(X[tr]), y[tr], sc.transform(X[te]), y[te]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LOSO 분할을 확인한다")
    parser.add_argument("--data", default=DATASET)
    args = parser.parse_args(argv)

    d = np.load(args.data)
    print(f"X {d['X'].shape} · 피험자 {len(np.unique(d['subject']))}명 → LOSO")
    for s, Xtr, ytr, Xte, yte in loso_folds(d):
        print(f"테스트 {s}: 학습 {Xtr.shape[0]}창 / 테스트 {Xte.shape[0]}창  "
              f"(학습 평균 {np.abs(Xtr.mean(0)).max():.0e}, 테스트 평균 최대 {np.abs(Xte.mean(0)).max():.2f})")


if __name__ == "__main__":
    main()

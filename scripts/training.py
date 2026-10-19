"""3주차 실습: LOSO 교차검증으로 모델 학습·평가

실행: python scripts/training.py [--data outputs/dataset.npz]

dataset.npz를 읽어 loso_folds()로 피험자 단위 8-fold를 돌리고, fold(= 테스트 피험자)별 점수와
평균 ± 표준편차를 낸다. 분할·표준화는 scripts/loso.py, 특징 추출은 scripts/preprocessing.py.
"""
import argparse

import numpy as np

from loso import DATASET, loso_folds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LOSO 교차검증으로 모델을 학습·평가한다")
    parser.add_argument("--data", default=DATASET)
    args = parser.parse_args(argv)

    d = np.load(args.data)
    for s, Xtr, ytr, Xte, yte in loso_folds(d):
        pass  # 3주차: 여기에 모델 fit(Xtr, ytr) → score(Xte, yte), fold별 점수 모아 평균 ± 표준편차


if __name__ == "__main__":
    main()

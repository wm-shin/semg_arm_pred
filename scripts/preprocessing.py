"""2주차 실습: 전처리와 특징 추출 → dataset.npz

실행: python scripts/preprocessing.py [--out outputs/dataset.npz] [--test-size 0.2] [--seed 42] [--no-filter]

시행 하나 (5000, 5) → 필터 → 동작 구간 2초 (2000, 5) → 200 ms 창 19개 (19, 200, 5)
→ 창마다 25차원 특징 (19, 25). 시행 단위로 학습/테스트를 나눈 뒤 학습셋 통계로만 표준화한다.
흐름도: docs/preprocessing.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import CH_NAMES, FS, ROOT, load_trials

NOTCH_HZ, NOTCH_Q = 60.0, 30.0          # 전원선 잡음 노치 (Q는 실습 설정)
BAND_HZ, BAND_ORDER = (20.0, 450.0), 4  # 근활성 대역, 4차 Butterworth
START, LEN = 1000, 2000                 # 동작 구간: mark 후 1.0~3.0초 (1주차 4-2에서 확인)
WIN, HOP = 200, 100                     # 200 ms 창을 100 ms씩 밀기 (50% 겹침)
ZC_TH = 0.01                            # ZC 계산 시 잡음의 미세한 부호 변화를 무시하는 임계값
FEATURE_NAMES = [f"{f}_{ch}" for f in ("RMS", "MAV", "WL", "ZC", "SSC") for ch in CH_NAMES]


# ---- 단계 1~4: 시행 하나를 처리하는 함수들 ----------------------------------------------

def preprocess(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """60 Hz 노치 + 20~450 Hz 대역통과. x: (샘플, 채널). filtfilt라 신호가 밀리지 않는다."""
    bn, an = iirnotch(NOTCH_HZ, NOTCH_Q, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(BAND_ORDER, [BAND_HZ[0] / (fs / 2), BAND_HZ[1] / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def cut(x: np.ndarray, start: int = START, n: int = LEN) -> np.ndarray:
    """시행에서 분석 구간만 잘라낸다. (5000, 5) -> (2000, 5)"""
    return x[start:start + n]


def make_windows(x: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    """슬라이딩 윈도우. (2000, 5) -> (19, 200, 5). 창 수 = (len - win) // hop + 1"""
    n = (len(x) - win) // hop + 1
    return np.stack([x[i * hop:i * hop + win] for i in range(n)])


def td_features(w: np.ndarray, th: float = ZC_TH) -> np.ndarray:
    """창 하나의 시간영역 특징 5종 x 5채널. (200, 5) -> (25,)  순서는 FEATURE_NAMES와 같다."""
    rms = np.sqrt(np.mean(w ** 2, axis=0))                     # 제곱 평균의 제곱근: 신호 세기
    mav = np.mean(np.abs(w), axis=0)                           # 절댓값 평균
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)                             # 인접 샘플 차이의 누적: 파형 복잡도
    zc = np.sum(((w[:-1] * w[1:]) < 0) & (np.abs(d) >= th), axis=0)  # 0을 지난 횟수 (작은 흔들림은 무시)
    ssc = np.sum((d[:-1] * d[1:]) < 0, axis=0)                 # 기울기 부호가 바뀐 횟수
    return np.concatenate([rms, mav, wl, zc, ssc])


def trial_features(x: np.ndarray, do_filter: bool = True) -> np.ndarray:
    """시행 하나를 끝까지 처리한다. (5000, 5) -> (19, 25)"""
    if do_filter:
        x = preprocess(x)
    return np.stack([td_features(w) for w in make_windows(cut(x))])


# ---- 단계 5~6: 데이터셋 만들기 ------------------------------------------------------------

def build(X_all: np.ndarray, labels_all: np.ndarray, idx: np.ndarray,
          do_filter: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """idx로 고른 시행들을 창 단위 특징 데이터셋으로 만든다.

    학습/테스트를 시행 단위로 먼저 나눈 뒤 각 집합에 따로 호출한다 (같은 시행의 겹치는 창이
    양쪽에 섞이는 데이터 누수를 막기 위해).

    Returns:
        X: (len(idx) x 19, 25) 특징
        y: (len(idx) x 19,)    창마다의 라벨
        g: (len(idx) x 19,)    창이 어느 시행(원래 행 번호)에서 왔는지. 교차검증에서 묶는 데 쓴다
    """
    feats = np.stack([trial_features(X_all[i], do_filter) for i in idx])  # (n, 19, 25)
    n, k, d = feats.shape
    return (feats.reshape(n * k, d),
            np.repeat(np.asarray(labels_all).astype(str)[idx], k),  # str로 바꿔야 np.load 때 pickle이 필요 없다
            np.repeat(np.asarray(idx), k))


def make_dataset(X_all: np.ndarray, trials: pd.DataFrame, test_size: float = 0.2, seed: int = 42,
                 do_filter: bool = True) -> dict[str, np.ndarray]:
    """시행 단위로 분할 → 각 집합에서 특징 → 학습셋 통계로만 표준화.

    Returns: np.savez(**결과)로 바로 저장할 수 있는 dict
        Xtr, Xte: 표준화된 특징 (창 수, 25) float32
        ytr, yte: 창마다의 라벨
        gtr, gte: 창이 온 시행 번호 (교차검증에서 같은 시행이 갈라지지 않게 묶는 용도)
        subject_tr, subject_te: 창이 온 피험자
        feature_names: 25개 열 이름
    """
    labels = trials["label"].to_numpy()
    idx_tr, idx_te = train_test_split(trials.index.to_numpy(), test_size=test_size,
                                      stratify=labels, random_state=seed)   # 시행 단위로 먼저 나눈다
    Xtr, ytr, gtr = build(X_all, labels, idx_tr, do_filter)
    Xte, yte, gte = build(X_all, labels, idx_te, do_filter)
    sc = StandardScaler().fit(Xtr)                                            # 통계는 학습셋에서만
    subject = trials["subject"].to_numpy().astype(str)
    return dict(Xtr=sc.transform(Xtr).astype(np.float32), ytr=ytr, gtr=gtr,
                Xte=sc.transform(Xte).astype(np.float32), yte=yte, gte=gte,
                subject_tr=subject[gtr], subject_te=subject[gte],
                feature_names=np.array(FEATURE_NAMES))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="sEMG 시행을 25차원 특징 데이터셋(npz)으로 만든다")
    parser.add_argument("--out", default=ROOT / "outputs" / "dataset.npz")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-filter", action="store_true", help="필터를 다시 걸지 않는다 (CSV는 이미 필터된 신호)")
    args = parser.parse_args(argv)

    X, trials = load_trials()
    d = make_dataset(X, trials, args.test_size, args.seed, do_filter=not args.no_filter)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)

    print(f"시행 {len(trials)}개 → 학습 {len(np.unique(d['gtr']))}회 / 테스트 {len(np.unique(d['gte']))}회 (시행 단위 분할)")
    print(f"Xtr {d['Xtr'].shape} / Xte {d['Xte'].shape}  (창 수, 25차원)")
    print(f"학습셋 평균 최대 절댓값 {np.abs(d['Xtr'].mean(0)).max():.1e}, 표준편차 {d['Xtr'].std(0).min():.3f}~{d['Xtr'].std(0).max():.3f}")
    print("\n클래스별 창 수:")
    print(pd.DataFrame({"학습": pd.Series(d["ytr"]).value_counts(),
                        "테스트": pd.Series(d["yte"]).value_counts()}).sort_index().to_string())
    print(f"\n저장: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

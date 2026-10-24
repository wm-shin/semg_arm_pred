"""2주차 실습: 전처리와 특징 추출 → dataset.npz

[1단계] 결과물: outputs/dataset.npz. 2_training.py 전에 한 번 실행한다.

실행: python scripts/1_preprocessing.py [--out outputs/dataset.npz] [--no-filter] [--start 1.0]

시행 하나 (5000, 5) → 필터 → 동작 구간 2초 (2000, 5) → 200 ms 창 19개 (19, 200, 5)
→ 창마다 25차원 특징 (19, 25). 특징은 시행마다 따로 계산되어 분할과 무관하므로 전체 시행을
한 번에 만들어 표준화 전 상태로 저장한다. 학습/테스트 분할(피험자 단위 LOSO)과 표준화는
fold마다 달라지므로 scripts/loso.py 에서 한다.
흐름도: mermaid/preprocessing.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

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


def trial_features(x: np.ndarray, do_filter: bool = True, start: int = START) -> np.ndarray:
    """시행 하나를 끝까지 처리한다. (5000, 5) -> (19, 25)"""
    if do_filter:
        x = preprocess(x)
    return np.stack([td_features(w) for w in make_windows(cut(x, start))])


# ---- 단계 5: 데이터셋 만들기 ----------------------------------------------------------------

def make_dataset(X_all: np.ndarray, trials: pd.DataFrame,
                 do_filter: bool = True, start: int = START) -> dict[str, np.ndarray]:
    """모든 시행을 창 단위 특징 데이터셋으로 만든다. 표준화는 하지 않는다 (loso.py에서 fold마다).

    Returns: np.savez(**결과)로 바로 저장할 수 있는 dict. 행 하나 = 창 하나 (시행 수 x 19)
        X:       표준화 전 특징 (창 수, 25) float32
        y:       창마다의 라벨
        subject: 창이 온 피험자. LOSO에서 groups로 쓴다
        trial:   창이 온 시행 (trials 행 번호). 같은 시행의 겹치는 창을 묶을 때 쓴다
        rep:     창이 온 시행의 반복 회차 1~30. 피험자 안에서 회차 단위로 나눌 때 쓴다
        feature_names: 25개 열 이름
    """
    feats = np.stack([trial_features(x, do_filter, start) for x in X_all])  # (시행 수, 19, 25)
    n, k, d = feats.shape
    col = lambda c: np.repeat(trials[c].to_numpy(), k)               # 시행 정보를 창 19개에 복사
    return dict(X=feats.reshape(n * k, d).astype(np.float32),
                y=col("label").astype(str),        # str로 바꿔야 np.load 때 pickle이 필요 없다
                subject=col("subject").astype(str),
                trial=np.repeat(np.arange(n), k),
                rep=col("rep"),
                feature_names=np.array(FEATURE_NAMES))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="sEMG 시행을 25차원 특징 데이터셋(npz)으로 만든다")
    parser.add_argument("--out", default=ROOT / "outputs" / "dataset.npz")
    parser.add_argument("--no-filter", action="store_true", help="필터를 다시 걸지 않는다 (CSV는 이미 필터된 신호)")
    parser.add_argument("--start", type=float, default=START / FS, help="2초 세그먼트 시작 시각(초). 기본 1.0 (1주차 동작 구간). 1.5 = 강의 실습 가정")
    args = parser.parse_args(argv)

    X, trials = load_trials()
    d = make_dataset(X, trials, do_filter=not args.no_filter, start=int(args.start * FS))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)

    print(f"시행 {len(trials)}개 → X {d['X'].shape}  (창 수, 25차원, 표준화 전)")
    print(f"피험자 {len(np.unique(d['subject']))}명 (분할·표준화는 scripts/loso.py 에서)")
    print("\n피험자 x 클래스별 창 수:")
    print(pd.crosstab(pd.Series(d["subject"], name="subject"), pd.Series(d["y"], name="label")).to_string())
    print(f"\n저장: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

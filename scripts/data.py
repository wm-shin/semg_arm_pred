"""sEMG Hand Gesture · Elbow Angle 데이터셋 읽기 (1주차 실습 3-1, 3-3).

실행:   python scripts/data.py          → 시행 수 · 배열 모양 · mark 간격 · 클래스별 분포 출력
재사용: from data import load_trials    (scripts/preprocessing.py 가 이렇게 쓴다)

CSV 64개 = 피험자 8명 x 각도 2종 x 동작 4종. 파일 하나에 같은 클래스의 30회 시행이
이어서 기록되어 있고, mark=1인 샘플이 각 시행(5초)의 시작이다.
흐름도: docs/data.md
"""
import argparse
import glob
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 폴더. 어느 위치에서 실행해도 같은 경로를 가리킨다
DATA_DIR = ROOT / "data" / "data"
FS = 1000                      # 샘플링 주파수 (Hz)
TRIAL_LEN = 5 * FS             # 시행 하나 = 5초 (휴식 2 + 동작 1 + 휴식 2)
CH_COLS = [f"bandpass {i}" for i in range(1, 6)]
CH_NAMES = CH_COLS   # 그림·특징 이름에도 CSV 열 이름을 그대로 쓴다
# README의 채널-근육 대응: 1 요측수근신근(ECR), 2 총지신근(ED), 3 천지굴근(FDS), 4 상완이두근(Biceps), 5 상완삼두근(Triceps)
# 파일 이름의 동작 표기 -> 라벨. extor는 회외(supination)
GESTURES = {"fist": "fist", "open": "open", "pro": "pronation", "extor": "supination"}


def list_files(data_dir: Path | str = DATA_DIR) -> list[str]:
    """data_dir 아래 CSV 경로를 정렬해서 돌려준다 (64개)."""
    return sorted(glob.glob(str(Path(data_dir) / "*.csv")))


def parse_name(path: str) -> tuple[str, int, str]:
    """파일 경로에서 (피험자, 각도, 동작)을 읽는다. 예: ".../a_30 fist.csv" -> ("a", 30, "fist")"""
    stem = Path(path).stem                  # "a_30 fist"
    subject, rest = stem.split("_")         # "a", "30 fist"
    angle, gesture = rest.split(" ")        # "30", "fist"
    return subject, int(angle), GESTURES[gesture]


def label_of(path: str) -> str:
    """클래스 라벨 = 각도_동작 (예: '30_fist'). 4동작 x 2각도 = 8클래스"""
    _, angle, gesture = parse_name(path)
    return f"{angle}_{gesture}"


def load_trials(data_dir: Path | str = DATA_DIR,
                trial_len: int = TRIAL_LEN) -> tuple[np.ndarray, pd.DataFrame]:
    """모든 CSV를 읽어 시행 단위로 자른다.

    Returns:
        X:      (시행 수, trial_len, 5) float32. X[i]가 i번째 시행의 5채널 신호
        trials: 시행 수만큼의 행. subject, angle, gesture, label, rep, start 열.
                trials의 i번째 행이 X[i]의 설명 (순서로 짝이 맞으므로 한쪽만 정렬하지 말 것)
    """
    rows, segments = [], []
    for f in list_files(data_dir):
        subject, angle, gesture = parse_name(f)
        df = pd.read_csv(f)
        emg = df[CH_COLS].to_numpy(np.float32)
        starts = np.flatnonzero(df["mark"].to_numpy() == 1)   # 시행 시작점 30개
        for rep, s in enumerate(starts, 1):
            segments.append(emg[s:s + trial_len])                # mark부터 5초
            rows.append({"subject": subject, "angle": angle, "gesture": gesture,
                         "label": label_of(f), "rep": rep, "start": int(s)})
    return np.stack(segments), pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="sEMG CSV를 시행 단위로 읽어 구조를 확인한다")
    parser.add_argument("--data-dir", default=DATA_DIR, help=f"CSV 폴더 (기본 {DATA_DIR})")
    args = parser.parse_args(argv)

    X, trials = load_trials(args.data_dir)
    gaps = trials.groupby(["subject", "label"])["start"].diff().dropna().astype(int)
    print("파일 수:", len(list_files(args.data_dir)))
    print("총 시행 수:", len(trials), "(기대값 8명 x 8클래스 x 30회 = 1920)")
    print("시행 배열 모양:", X.shape, "(시행 수, 샘플 수, 채널 수)")
    print("mark 간격 (샘플):", dict(Counter(gaps)))
    print("\n피험자 x 클래스별 시행 수:")
    print(pd.crosstab(trials["subject"], trials["label"]).to_string())


if __name__ == "__main__":
    main()

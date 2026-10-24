# scripts/ 실행 순서

번호가 붙은 파일만 순서대로 실행하면 결과물이 나온다. 번호 없는 파일은 그 스크립트들이 import해서 쓰는 모듈이다 (파이썬은 숫자로 시작하는 모듈을 import할 수 없어서 번호를 못 붙인다).

| 실행 | 파일 | 결과물 | 설명 |
|---|---|---|---|
| 1 | `python scripts/1_preprocessing.py` | `outputs/dataset.npz` | 필터 → 2초 자르기 → 200 ms 창 → 25차원 특징. [docs/preprocessing.md](../docs/preprocessing.md) |
| 2 | `python scripts/2_training.py --split loso` | `outputs/week_3/loso/`, `figures/week_3/confusion8_loso.png` | SVM · RF · XGB 교차검증 + t-검정 + 혼동행렬. [docs/training.md](../docs/training.md) |
| 2 | `python scripts/2_training.py --split trialwise` | `outputs/week_3/trialwise/`, `figures/week_3/confusion8_trialwise.png` | 논문·강의 방식(시행 단위 5-fold) 대조용 |
| 3 | `python scripts/3_rf_validation.py --split loso` (또는 trialwise) | `outputs/week_3/{split}/rf_val_*.csv`, `figures/week_3/rf_val_{split}.png` | RF 검증 히스토리: 트리 수·깊이에 따른 검증 정확도 (fold 평균 ± 표준편차). 2단계의 RF_TREES·RF_MAX_DEPTH 근거. [docs/rf_validation.md](../docs/rf_validation.md) |
| 4 | `python scripts/4_cut_sweep.py` | `outputs/week_3/cut_sweep.csv`, `figures/week_3/cut_sweep.png` | cut 시작점을 0~3 s로 옮겨 RF trial-wise·LOSO 정확도 비교. 휴식 구간에서 trial-wise가 오르고 LOSO가 우연으로 가는 것 = 파일별 잡음 서명의 증거. [docs/cut_sweep.md](../docs/cut_sweep.md) |

| 모듈 (import용) | 쓰는 곳 | 직접 실행하면 |
|---|---|---|
| `data.py` | 1_preprocessing.py | CSV 64개 읽어 시행 수 · 배열 모양 · 클래스 분포 출력. [docs/data.md](../docs/data.md) |
| `loso.py` | 2_training.py | fold별 학습/테스트 창 수와 표준화 확인 출력 (`--split loso|trialwise`). [docs/loso.md](../docs/loso.md) |

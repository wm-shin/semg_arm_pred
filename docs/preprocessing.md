# scripts/1_preprocessing.py — 전처리 · 특징 추출 · 데이터셋 저장

실행: `python scripts/1_preprocessing.py [--out outputs/dataset.npz] [--no-filter]`

시행 1920개 전체에 필터 → 2초 자르기 → 200 ms 윈도우 → 25차원 특징을 만들어 `dataset.npz`로 저장한다. 학습/테스트 분할과 표준화는 하지 않는다 (`scripts/loso.py`가 fold마다 한다).

```mermaid
flowchart TD
    start((시작)) --> M["main()<br/>--out · --no-filter"]
    M --> L["load_trials()  scripts/data.py<br/>→ X (1920, 5000, 5) · trials 표"]
    L --> MD["make_dataset(X, trials)"]

    subgraph P["trial_features(x): 시행 1920개마다 반복"]
        direction TB
        F["preprocess()<br/>60 Hz 노치 + 20~450 Hz 대역통과<br/>(5000, 5) → (5000, 5)"]
        C["cut()<br/>동작 구간 2초 (1.0~3.0 s)<br/>→ (2000, 5)"]
        W["make_windows()<br/>200 ms 창 · 100 ms 간격<br/>→ (19, 200, 5)"]
        T["td_features() x 19<br/>RMS·MAV·WL·ZC·SSC x 5채널<br/>→ (19, 25)"]
        F --> C --> W --> T
    end

    MD --> P
    P --> NPZ[("np.savez(out)<br/>X (1920x19, 25) · y · subject · trial · rep · feature_names")]
    NPZ --> OUT["main(): 확인 출력<br/>모양 · 피험자 x 클래스별 창 수"]
    OUT --> stop(((끝)))
```

## 출력 `dataset.npz`

numpy 배열 6개를 묶은 zip. 행 하나 = 창 하나이고, 길이 36480(= 시행 1920 x 창 19)인 배열들은 i번째 행이 서로 짝이 맞는다.

| 키 | 모양 | 내용 |
|---|---|---|
| `X` | (36480, 25) | 표준화 전 특징 |
| `y` | (36480,) | 라벨 `'30_fist'` ... 8종 |
| `subject` | (36480,) | 피험자 `'a'`~`'h'`. 피험자마다 4560행 (8클래스 x 30회 x 19창). LOSO 분할 기준 |
| `trial` | (36480,) | 시행 번호 0~1919 |
| `rep` | (36480,) | 반복 회차 1~30 |
| `feature_names` | (25,) | `X`의 열 이름 |

행 순서: 피험자 a → 클래스 8개 → 회차 1~30 → 창 1~19, 그 다음 피험자 b.

## 왜 분할·표준화를 여기서 안 하나

필터·자르기·윈도우·특징은 시행 하나 안에서만 계산되므로 어떻게 나누든 결과가 같다 → 한 번만 만들어 저장한다. 표준화의 평균·표준편차는 학습셋에서만 구해야 하는데 LOSO에서는 학습셋이 fold마다 바뀐다 → 저장하지 않고 fold 안에서 한다.

## 설정 메모 (강의 예시와 다른 점)

| 항목 | 강의 예시 | 이 스크립트 |
|---|---|---|
| `cut()` 시작점 | `START=1500` (1.5 s, 실습 가정) | 1주차 4-2에서 확인한 실제 동작 구간 1.0~2.7 s에 맞춰 `START=1000` |
| 필터 | 원신호에 적용 | CSV 열 이름이 `bandpass 1~5`로, 이미 같은 필터가 걸려 있음. 강의 순서대로 한 번 더 걸되(차이 미미) `--no-filter`로 끌 수 있음 |
| ZC 임계값 | `th=0.01` | 같은 값. 단, 이 데이터는 진폭이 작아 Biceps·Triceps의 ZC가 0인 창이 30~50% → 3주차에서 낮춘 값과 비교해 볼 것 |

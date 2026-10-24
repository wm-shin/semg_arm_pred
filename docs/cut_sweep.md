# scripts/4_cut_sweep.py — cut 위치 실험: 모델이 동작을 보는가, 파일별 잡음을 보는가

실행: `python scripts/4_cut_sweep.py [--starts 0 0.5 1 1.5 2 2.5 3]`

`1_preprocessing.py`의 cut 시작점(기본 1.0 s)을 0~3 s로 옮겨 가며 같은 창·특징을 메모리에서 다시 만들고, RF 하나로 trial-wise 5-fold와 LOSO 8-fold 정확도를 잰다. 표본은 `2_training.py`와 같은 시행 단위(475차원). `dataset.npz`는 건드리지 않는다. 3주차 리포트에서 "trial-wise 점수에는 근육 활동 외에 파일별 잡음 서명이 섞여 있고, LOSO가 그것을 걷어낸 값이다"를 보이는 핵심 실험이다.

```mermaid
flowchart TD
    start((시작)) --> M["main()<br/>--starts 0 0.5 … 3"]
    M --> L["load_trials()  scripts/data.py<br/>→ X_all (1920, 5000, 5)"]
    L --> F["preprocess() x 1920  1_preprocessing.py<br/>필터는 한 번만"]
    F --> ENV["시행 평균 RMS 포락선<br/>100 ms 창 · 채널 평균 · 0~5 s<br/>(동작 위치 표시용)"]

    subgraph G["cut 시작점 반복"]
        direction TB
        C["features_at(start)<br/>cut(x, start) → make_windows → td_features → to_trials<br/>1_preprocessing.py · loso.py 와 같은 함수, 시작점만 다름<br/>→ d = {X (1920, 475), y, subject, trial}"]
        TW["rf_cv(d, 'trialwise')<br/>FOLDS['trialwise'] → RF 100트리 x 5 fold<br/>fold 안 표준화는 loso.py 가 한다"]
        LO["rf_cv(d, 'loso')<br/>FOLDS['loso'] → RF x 8 fold"]
        R["행 1개: start · 구간 RMS ·<br/>trial-wise 평균±std · LOSO 평균±std"]
        C --> TW --> R
        C --> LO --> R
    end
    F --> G

    R --> CSV[("outputs/week_3/cut_sweep.csv<br/>시작점당 1행")]
    CSV --> P["plot()<br/>위: RMS 포락선 + cut 구간 음영<br/>아래: 시작점별 trial-wise · LOSO · 우연선"]
    ENV --> P
    P --> PNG[("figures/week_3/cut_sweep.png")]
    PNG --> stop(((끝)))

    style C stroke:#3fb9c4,stroke-width:3px
    style R stroke:#3fb9c4,stroke-width:3px
```

## 그림에서 봐야 할 것

- **특징 함수는 1단계 것을 그대로 쓴다.** `cut()`의 시작점만 다르고 필터·창·특징은 같으므로, 점수 차이는 오직 "시행의 어느 2초를 보느냐"에서 온다.
- **시행 단위 묶기와 분할·표준화는 2단계와 같은 함수를 쓴다.** 그래서 start 1.0 s 행의 값은 `2_training.py`의 RF 결과(trial-wise 0.968, LOSO 0.472)와 일치해야 하고, 실제로 일치한다.
- **RMS 포락선이 "동작이 어디 있는지"의 근거다.** 1주차에서 확인한 동작 구간 1.0~2.7 s가 포락선의 봉우리로 다시 보인다. 2.5 s 이후는 휴식이다.

## 결과 (2026-10-24, RF 100트리)

| cut 구간 | 내용 | 구간 RMS (x10⁻³) | trial-wise | LOSO |
|---|---|---|---|---|
| 0.0~2.0 s | 휴식 + 동작 앞부분 | 37.2 | 0.966 | 0.439 |
| 0.5~2.5 s | 동작 대부분 | 40.4 | 0.970 | **0.493** |
| **1.0~3.0 s** | **동작 구간 (기본)** | **40.5** | **0.968** | 0.473 |
| 1.5~3.5 s | 동작 뒷부분 + 휴식 | 36.6 | 0.962 | 0.418 |
| 2.0~4.0 s | 동작 끝 + 휴식 | 18.5 | 0.888 | 0.154 |
| 2.5~4.5 s | 거의 휴식 | 9.3 | 0.801 | 0.105 |
| 3.0~5.0 s | 휴식 | 6.3 | 0.764 | 0.129 |

우연 수준 0.125. fold 표준편차는 `outputs/week_3/cut_sweep.csv`.

## 결과 읽기

- **LOSO는 동작을 따라간다.** 동작이 있는 구간(0.5~3.0 s)에서만 우연을 넘고, 휴식 구간(2.5 s 이후)에서는 우연(0.10~0.15)으로 떨어진다. LOSO가 재는 것은 근육 활동이고, 그 외에는 아무것도 못 쓴다.
- **trial-wise는 휴식 구간에서도 높다.** 근육 활동이 없는 3~5 s에서 0.764다. 이 데이터는 피험자 x 클래스마다 CSV 파일 하나가 별도 세션이라 파일마다 휴식 잡음 수준이 다르고(`docs/training.md`), 같은 사람의 같은 파일이 학습에 있으면 "어느 파일에서 온 표본인가"로 클래스를 맞힐 수 있다. 이 서명은 trial-wise 점수에 약 0.76의 바닥을 깔아 준다.
- **동작 정보는 그 위에 20%p를 얹는다.** 동작 구간 0.968 - 휴식 구간 0.764. 논문의 96.8%는 "세션 서명 약 76% + 동작 정보 약 20%p"로 읽을 수 있다. 동작 정보만으로 처음 보는 사람을 맞히면 47%(LOSO)이고, 나머지는 같은 사람·같은 세션이라서 얻는 점수다.
- **논문의 trial-wise 96.7%에 대한 함의.** 논문 자신도 4절에서 "같은 참가자가 학습·테스트에 들어가므로 피험자 독립 일반화가 아니다"라고 썼다. 이 실험은 그 한계의 크기를 보여 준다. 리포트 한계 ⑦의 첫 항목으로 쓴다.
- **근본 원인은 데이터 수집 설계다.** 클래스마다 파일(세션)을 따로 기록했기 때문에 세션 서명과 클래스가 하나로 묶였다. 여덟 클래스를 한 세션에서 섞어 기록했다면 trial-wise도 동작 정보만 재는 평가가 됐을 것이다.

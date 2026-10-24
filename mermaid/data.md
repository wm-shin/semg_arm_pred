# scripts/data.py — CSV를 시행 단위로 읽기

실행: `python scripts/data.py` → 시행 수 · 배열 모양 · mark 간격 · 피험자x클래스 표 출력
다른 스크립트에서: `from data import load_trials`

CSV 64개(피험자 8명 x 각도 2종 x 동작 4종)를 읽어, 파일마다 `mark=1` 지점에서 5초(5000샘플)씩 30번 잘라 시행 1920개를 만든다.

```mermaid
flowchart TD
    start((시작)) --> M["main()<br/>--data-dir (기본 data/data)"]
    M --> L["load_trials(data_dir)"]
    L --> LF["list_files()<br/>→ CSV 경로 64개 (정렬)"]
    LF --> D1{"남은 파일이 있나?<br/>(64개 반복)"}
    D1 -- 있음 --> PN["parse_name(f)<br/>→ (subject, angle, gesture)<br/>예: ('a', 30, 'fist')"]
    PN --> RC["pd.read_csv(f)<br/>df: 약 151,000행 x 8열"]
    RC --> CH["df[CH_COLS] → float32<br/>emg: 약 151,000 x 5채널"]
    CH --> MK["mark == 1 위치 찾기<br/>starts: 30개, 5000샘플 간격"]
    MK --> D2{"남은 시작점 s가 있나?<br/>(30개 반복)"}
    D2 -- 있음 --> CUT["emg[s : s + TRIAL_LEN]<br/>시행 1개 (5000, 5) → segments"]
    CUT --> ROW["시행 정보 한 줄 → rows<br/>subject · angle · gesture · label · rep · start"]
    ROW -- 다음 s --> D2
    D2 -- "30개 끝 → 다음 파일" --> D1
    D1 -- "64개 모두 끝" --> ST["np.stack(segments), DataFrame(rows)<br/>X (1920, 5000, 5) · trials 1920행 x 6열"]
    ST --> OUT["main(): 확인 출력<br/>시행 수 · 모양 · mark 간격 · crosstab"]
    OUT --> stop(((끝)))

    style CUT stroke:#3fb9c4,stroke-width:3px
```

## 데이터 예시

원본 CSV 한 파일 (`a_30 extor.csv`, 151361행). 1 ms마다 한 행, `mark=1`이 시행 시작. 첫 시행이 70행에서 시작한다.

|  | bandpass 1 | bandpass 2 | bandpass 3 | bandpass 4 | bandpass 5 | time | onset | mark |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.0012 | -0.0016 | 0.0001 | 0.0032 | 0.0022 | 4.2480 | 0 | 0 |
| 1 | -0.0019 | -0.0023 | 0.0001 | 0.0028 | 0.0020 | 4.2490 | 0 | 0 |
| 70 | -0.0055 | -0.0000 | 0.0015 | 0.0037 | 0.0012 | 4.3180 | 1 | 1 |
| 71 | -0.0052 | -0.0004 | 0.0011 | 0.0026 | 0.0002 | 4.3190 | 0 | 0 |
| 5070 | -0.0092 | 0.0134 | 0.0015 | 0.0005 | -0.0026 | 9.3180 | 0 | 1 |

`load_trials()`가 돌려주는 `trials` 표 (1920행). i번째 행이 `X[i]`(5000 x 5 신호)의 설명이다. `start`는 그 시행이 CSV의 몇 번째 행에서 시작하는지.

|  | subject | angle | gesture | label | rep | start |
|---|---|---|---|---|---|---|
| 0 | a | 30 | supination | 30_supination | 1 | 70 |
| 1 | a | 30 | supination | 30_supination | 2 | 5070 |
| 29 | a | 30 | supination | 30_supination | 30 | 145070 |
| 30 | a | 30 | fist | 30_fist | 1 | 0 |
| 31 | a | 30 | fist | 30_fist | 2 | 5000 |
| 1919 | h | 60 | pronation | 60_pronation | 30 | 145000 |

`X.shape` = (1920, 5000, 5) (시행, 샘플 5초 x 1 kHz, 채널). `X[0][:3]`:

| bandpass 1 | bandpass 2 | bandpass 3 | bandpass 4 | bandpass 5 |
|---|---|---|---|---|
| -0.0055 | -0.0000 | 0.0015 | 0.0037 | 0.0012 |
| -0.0052 | -0.0004 | 0.0011 | 0.0026 | 0.0002 |
| -0.0052 | -0.0008 | 0.0008 | 0.0016 | -0.0012 |

## 메모

- `X[i]`와 `trials`의 `i`번째 행이 짝이다. 한쪽만 정렬하거나 섞으면 어긋난다.
- 첫 `mark` 앞의 대기 구간과 30번째 시행 뒤의 남은 기록은 어떤 조각에도 들어가지 않아 자동으로 빠진다.
- `mark` 간격이 4999~5001로 1샘플(1 ms)씩 흔들리는 파일이 있지만, 자르는 길이는 항상 5000으로 고정했다 (시행 길이가 같아야 배열로 쌓인다).
- `extor`는 회외(supination)로 바꿔 라벨에 쓴다. 라벨 = `각도_동작` (8클래스).

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

## 메모

- `X[i]`와 `trials`의 `i`번째 행이 짝이다. 한쪽만 정렬하거나 섞으면 어긋난다.
- 첫 `mark` 앞의 대기 구간과 30번째 시행 뒤의 남은 기록은 어떤 조각에도 들어가지 않아 자동으로 빠진다.
- `mark` 간격이 4999~5001로 1샘플(1 ms)씩 흔들리는 파일이 있지만, 자르는 길이는 항상 5000으로 고정했다 (시행 길이가 같아야 배열로 쌓인다).
- `extor`는 회외(supination)로 바꿔 라벨에 쓴다. 라벨 = `각도_동작` (8클래스).

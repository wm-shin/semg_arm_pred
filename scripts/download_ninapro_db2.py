"""NinaPro DB2 다운로드 스크립트.

공식 사이트는 피험자별 zip(DB2_s1.zip ~ DB2_s40.zip, 각 ~470MB)만 제공하므로
필요한 피험자만 골라 받고, 원하는 Exercise의 .mat 파일만 압축 해제한다.

사용 예:
    python scripts/download_ninapro_db2.py                     # 피험자 1~3, E1(Exercise B)
    python scripts/download_ninapro_db2.py --subjects 1 2 5
    python scripts/download_ninapro_db2.py --subjects 1-10 --exercises 1 2
    python scripts/download_ninapro_db2.py --keep-zip          # zip 원본 보관

결과:
    data/ninapro_db2/s1/S1_E1_A1.mat ...

인용: Atzori et al., "Electromyography data for non-invasive naturally-controlled
robotic hand prostheses", Scientific Data, 2014.
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{}.zip"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "ninapro_db2"


def parse_subjects(tokens):
    """'1 2 5' 또는 '1-10' 형태를 피험자 번호 리스트로 변환."""
    subjects = []
    for tok in tokens:
        if "-" in tok:
            lo, hi = map(int, tok.split("-"))
            subjects.extend(range(lo, hi + 1))
        else:
            subjects.append(int(tok))
    bad = [s for s in subjects if not 1 <= s <= 40]
    if bad:
        sys.exit(f"피험자 번호는 1~40이어야 한다: {bad}")
    return sorted(set(subjects))


def download(url, dest):
    """진행률을 표시하며 다운로드. 중간에 끊긴 파일(.part)은 이어받는다."""
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url)
    if done:
        req.add_header("Range", f"bytes={done}-")

    with urllib.request.urlopen(req) as resp:
        if done and resp.status != 206:  # 서버가 이어받기를 거부하면 처음부터
            done = 0
        total = done + int(resp.headers.get("Content-Length", 0))
        with open(part, "ab" if done else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
    print()
    part.rename(dest)


def extract(zip_path, out_dir, exercises):
    """zip 안에서 S*_E{n}_A1.mat 만 골라 out_dir 에 평탄하게 푼다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [f"_E{e}_" for e in exercises]
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if not name.endswith(".mat") or not any(w in name for w in wanted):
                continue
            with zf.open(info) as src, open(out_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)
    return extracted


def main():
    p = argparse.ArgumentParser(description="NinaPro DB2 피험자별 다운로드")
    p.add_argument("--subjects", nargs="+", default=["1-3"],
                   help="피험자 번호 (예: 1 2 5 또는 1-10). 기본 1-3")
    p.add_argument("--exercises", nargs="+", type=int, default=[1], choices=[1, 2, 3],
                   help="1=Exercise B(손동작·손목), 2=C(쥐기), 3=D(힘). 기본 1")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="저장 폴더")
    p.add_argument("--keep-zip", action="store_true", help="압축 해제 후 zip 보관")
    args = p.parse_args()

    subjects = parse_subjects(args.subjects)
    zip_dir = args.out / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    print(f"피험자 {subjects}, Exercise {args.exercises} → {args.out}")

    for s in subjects:
        sub_dir = args.out / f"s{s}"
        expected = [sub_dir / f"S{s}_E{e}_A1.mat" for e in args.exercises]
        if all(f.exists() for f in expected):
            print(f"[s{s}] 이미 있음, 건너뜀")
            continue

        zip_path = zip_dir / f"DB2_s{s}.zip"
        if not zip_path.exists():
            print(f"[s{s}] 다운로드 중...")
            download(BASE_URL.format(s), zip_path)

        names = extract(zip_path, sub_dir, args.exercises)
        print(f"[s{s}] 압축 해제: {', '.join(names) or '(해당 파일 없음)'}")
        if not args.keep_zip:
            zip_path.unlink()

    if not args.keep_zip and zip_dir.exists() and not any(zip_dir.iterdir()):
        zip_dir.rmdir()
    print("완료")


if __name__ == "__main__":
    main()

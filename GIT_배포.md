# GitHub 배포 — 남은 건 2단계뿐

로컬 준비는 **전부 끝났습니다.**

- git 저장소 초기화 ✅
- 20개 파일 커밋 완료 (2.4MB) ✅
- 브랜치 `main` 으로 설정 ✅
- 원격 주소 `https://github.com/jiwoo1105/video-grounding.git` 등록 ✅

이제 **GitHub에 빈 레포를 만들고 push** 하면 됩니다.

---

## 1단계 — GitHub에서 빈 레포 만들기

<https://github.com/new> 접속 후:

| 항목 | 값 |
|---|---|
| Repository name | **`video-grounding`** ← 정확히 이 이름 |
| 공개 범위 | **Public** ← Private 이면 인스턴스 clone 시 토큰 필요 |
| Add a README file | **체크 해제** |
| Add .gitignore | **None** |
| Choose a license | **None** |

**아래 3개를 반드시 체크 해제하세요.** 파일이 이미 있어서 충돌합니다.

`Create repository` 클릭.

---

## 2단계 — 맥 터미널에서 push

```bash
cd ~/Documents/Video_grounding
git push -u origin main
```

**이 한 줄이 전부입니다.** 원격 주소와 브랜치는 이미 설정돼 있습니다.

### 인증을 물어보면

`Username` 에 `jiwoo1105`, `Password` 에는 **계정 비밀번호가 아니라 토큰**을 넣어야 합니다.

1. <https://github.com/settings/tokens> → `Generate new token` → `Generate new token (classic)`
2. `Note` 에 아무 이름, `Expiration` 은 30 days 정도
3. **`repo` 체크박스만 선택**
4. `Generate token` → 나온 문자열 복사 (한 번만 보입니다)
5. 터미널 `Password` 자리에 붙여넣기 (화면에 안 보이는 게 정상)

> 맥에서 GitHub Desktop 이나 `gh` CLI 를 이미 쓰고 계셨다면 인증을 안 물어볼 수도 있습니다.

---

## 3단계 — 인스턴스에서 clone

엘리스 VSCode 터미널에서:

```bash
cd ~
rm -rf Video_grounding                       # 아까 만든 빈 폴더 정리
git clone https://github.com/jiwoo1105/video-grounding.git Video_grounding
cd Video_grounding
ls
```

`app.py`, `vtg_run.py`, `videos/2Y8XQ.mp4` 가 보이면 성공입니다.
**폴더 열기도 업로드도 필요 없습니다.**

---

## 4단계 — 실행

```bash
bash setup_env.sh                 # 10~15분. "지금 다운로드?" -> y
source ~/vtg-env/bin/activate
sudo apt-get install -y ffmpeg    # 없다고 나오면

python3 app.py                    # 웹 UI
```

터미널에 뜨는 포트 알림을 클릭하거나, 안 뜨면 `python3 app.py --share`.

이후는 `실행순서.md` 6번(사용법)부터 그대로입니다.

---

## 코드 수정 후 동기화

```bash
# 맥에서
git add -A && git commit -m "수정" && git push

# 인스턴스에서
cd ~/Video_grounding && git pull
```

---

## 레포에 들어간 것

| 파일 | 용도 |
|---|---|
| `app.py` | 웹 UI (모델 4종 드롭다운) |
| `demo.py` | UI 단일파일 버전 (백업용) |
| `vtg_run.py` | 단발 추론 CLI |
| `run_experiments.py` | 일괄 실험 + 리포트 |
| `setup_env.sh` | 환경 구성 |
| `test_vtg_run.py` / `test_e2e.py` | GPU 없이 검증 (124개) |
| `videos/2Y8XQ.mp4` | 데모 영상 2.2MB |
| `videos/external.json` | 질의 6개 + 정답 시각 |
| 문서 9개 | 실행순서, 영상분석, 논문정리 등 |

`.gitignore` 로 제외: `vtg_bundle.tar.gz`, `videos/_bench/`(수 GB), `results/`, `__pycache__/`

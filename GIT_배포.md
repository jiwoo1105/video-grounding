# GitHub 로 올리고 인스턴스에서 clone 하기

로컬 폴더는 **이미 git 저장소로 초기화되고 커밋까지 끝나 있습니다.**
GitHub에 레포만 만들어 push 하면 됩니다.

---

## 1. GitHub 레포 만들기

### 방법 A — 웹에서 (가장 확실)

1. <https://github.com/new> 접속
2. `Repository name` 에 **`video-grounding`** 입력
3. **Public** 선택 ← 중요. Private 이면 인스턴스에서 clone 할 때 토큰이 필요합니다
4. **README, .gitignore, license 는 전부 체크 해제** (이미 있어서 충돌합니다)
5. `Create repository`

### 방법 B — gh CLI 가 있다면 한 줄

```bash
cd ~/Documents/Video_grounding
gh repo create video-grounding --public --source=. --push
```

이러면 3번(push)까지 한 번에 끝납니다.

---

## 2. 맥에서 push

방법 A로 만드셨다면, 맥 터미널에서:

```bash
cd ~/Documents/Video_grounding

git remote add origin https://github.com/<내아이디>/video-grounding.git
git branch -M main
git push -u origin main
```

`<내아이디>` 를 본인 GitHub 아이디로 바꾸세요.

> 비밀번호를 물으면 **GitHub 계정 비밀번호가 아니라 Personal Access Token** 이 필요합니다.
> <https://github.com/settings/tokens> → `Generate new token (classic)` → `repo` 체크 → 생성 후 그 값을 붙여넣기.

---

## 3. 인스턴스에서 clone

엘리스 VSCode 터미널에서:

```bash
cd ~
git clone https://github.com/<내아이디>/video-grounding.git Video_grounding
cd Video_grounding
ls
```

`app.py`, `vtg_run.py`, `videos/2Y8XQ.mp4` 가 보이면 성공입니다.
**영상(2.2MB)도 레포에 포함돼 있어서 따로 받을 필요 없습니다.**

VSCode에서 `File` → `Open Folder` → `/home/elicer/Video_grounding` 을 열면
편집기에서도 파일이 보입니다. (안 열어도 터미널로 다 됩니다)

---

## 4. 이어서 실행

```bash
bash setup_env.sh                 # 10~15분. "y" 누르면 모델도 미리 받음
source ~/vtg-env/bin/activate

python3 app.py                    # 웹 UI
```

이후는 `실행순서.md` 4번부터 그대로입니다.

---

## 코드를 고쳤을 때

맥에서 고치고 push → 인스턴스에서 pull:

```bash
# 맥
git add -A && git commit -m "수정" && git push

# 인스턴스
cd ~/Video_grounding && git pull
```

---

## 레포에 들어간 것 / 안 들어간 것

| 포함 | 제외 (`.gitignore`) |
|---|---|
| 코드 6개 (`app.py`, `demo.py`, `vtg_run.py` 등) | `vtg_bundle.tar.gz` |
| 문서 8개 | `videos/_bench/` (수 GB 벤치마크) |
| `videos/2Y8XQ.mp4` (2.2MB 데모 영상) | `results/`, `__pycache__/` |
| `videos/external.json` (질의 + 정답) | 내가 추가한 다른 영상 |

총 **2.4MB** 라 push/clone 모두 몇 초면 끝납니다.

> 내 영상을 레포에 넣고 싶다면 `.gitignore` 의 `videos/*.mp4` 줄을 지우거나,
> `git add -f videos/내영상.mp4` 로 강제 추가하세요.

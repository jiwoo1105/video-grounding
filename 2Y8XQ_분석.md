# 2Y8XQ.mp4 — 영상 내용과 질의

프레임을 1초 간격으로 뽑아 눈으로 확인한 결과입니다.
**공식 정답(GT)이 아니라 제가 판단한 값**이라 ±1초 오차가 있을 수 있습니다.

---

## 영상 정보

| 항목 | 값 |
|---|---|
| 길이 | 30.77초 |
| 해상도 | 480 × 720 (세로) |
| 프레임레이트 | 6.12 fps |
| 용량 | 2.26 MB |

**내용**: 좁은 화장실. 검은 모자와 검은 티셔츠를 입은 남자가 변기에 앉아 있음.
무릎 위에 **분홍색 수건**, 바닥에 **투명 물병**이 놓여 있음.

---

## 타임라인

| 시간 | 일어나는 일 |
|---|---|
| **0 ~ 3초** | 분홍 수건을 손에 들고 펼치며 털어냄 |
| 3 ~ 4초 | 수건을 무릎에 놓고 앉음 |
| **4 ~ 6초** | 몸을 숙여 **바닥에서 물병을 집어 듦** |
| 6 ~ 16초 | 물병을 손에 들고 앉아 있음. 여러 번 몸을 숙임, 뚜껑을 만지작거림 |
| **17 ~ 19초** | **물병을 입에 대고 물을 마심** ← 공식 데모 질의가 가리키는 구간 |
| 20 ~ 25초 | 뚜껑 닫은 물병을 무릎 사이에 들고 앉아 있음 |
| **26 ~ 28초** | **분홍 수건을 들어 올려 얼굴을 닦음** |
| 29 ~ 30초 | 수건을 내리고 물병을 든 채 앉아 있음 |

---

## 쓸 질의 6개

UI에서 영상을 고르면 **드롭다운에 그대로 뜹니다.** 클릭만 하면 됩니다.

| # | 질의 | 정답 | 용도 |
|---|---|---|---|
| 1 | `A man drinks water with a glass` | 17~19초 | **공식 데모 질의** |
| 2 | `a man drinks from a plastic water bottle` | 17~19초 | 같은 사건, 정확한 표현 |
| 3 | `a man picks up a bottle from the floor` | 4~6초 | 다른 사건 |
| 4 | `a man wipes his face with a pink towel` | 26~28초 | 다른 사건 |
| 5 | `a man shakes out a pink towel` | 0~3초 | 영상 앞부분 |
| 6 | `a dog runs across the room` | **없음** | 한계 시연 |

---

## 발표 시연 순서

### ① 공식 질의로 시작

```
A man drinks water with a glass          → 정답 17~19초
```

잘 맞으면 tIoU 0.5 이상이 나옵니다.

### ② 여기서 흥미로운 지점 — 주석이 틀렸습니다

**영상에 나오는 건 유리컵(glass)이 아니라 플라스틱 물병입니다.**
공식 데모 질의 자체가 부정확해요.

```
A man drinks water with a glass            (부정확한 원본 표현)
a man drinks from a plastic water bottle   (실제 영상에 맞는 표현)
```

두 결과를 비교해 보세요. **이게 TimeLens 논문이 지적한 바로 그 문제입니다** —
Charades-STA 의 34.9% 가 "annotation accuracy issues" 를 갖고 있다는 것.
데모 영상에서도 그대로 드러납니다.

> "이 분야 표준 데이터셋의 라벨이 이 정도로 부정확했고,
>  그래서 TimeLens 팀이 9,404개 주석 중 6,463개의 질의를 다시 썼다"
> 로 자연스럽게 연결됩니다.

### ③ 다른 사건도 찾는지

```
a man picks up a bottle from the floor    → 정답 4~6초
a man wipes his face with a pink towel    → 정답 26~28초
```

**앞부분·중간·뒷부분 사건을 각각 맞히면** 영상 전체를 보고 있다는 뜻입니다.
한쪽에만 몰려 답하면 그것대로 관찰 거리고요.

### ④ 없는 사건 — 한계

```
a dog runs across the room                → 영상에 개가 없음
```

모델은 "없다"고 답하지 못하고 **아무 구간이나 뱉습니다.**
현재 이 분야의 미해결 과제(no-target rejection)입니다.

### ⑤ 모델을 바꿔서 비교

모델 드롭다운만 바꾸고 ①을 다시 돌립니다.

| 모델 | 백본 | 논문 Charades mIoU |
|---|---|---|
| **TimeLens-8B** | Qwen3-VL-8B | **55.2** |
| TimeLens-7B | Qwen2.5-VL-7B | 48.8 |
| TimeLens2-8B | Qwen3-VL-8B | 58.6 |

- **TimeLens-8B vs TimeLens-7B** — 백본 세대 차이 (Qwen3-VL vs Qwen2.5-VL)
- **TimeLens vs TimeLens2** — 같은 백본, 학습 방법 차이

> 학습 전 원본(Qwen3-VL-8B)과 비교하고 싶다면 UI 말고 터미널에서:
> ```bash
> python3 vtg_run.py --model qwen3-vl-8b --video videos/2Y8XQ.mp4 \
>     --query "A man drinks water with a glass"
> ```
> 17GB를 새로 받아야 하므로 발표 전에 미리 한 번 돌려두세요.

---

## 참고 — 프레임을 직접 보고 싶다면

```bash
ffmpeg -i videos/2Y8XQ.mp4 -vf "fps=1,scale=260:-2,\
drawtext=text='%{eif\:t\:d}s':x=6:y=6:fontsize=26:fontcolor=yellow:\
box=1:boxcolor=black@0.6,tile=4x4" -frames:v 1 sheet.jpg
```

1초 간격 프레임을 격자로 붙여 한 장에 담습니다. 시간이 노란 글씨로 찍혀요.

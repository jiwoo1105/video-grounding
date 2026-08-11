# 영상 다운로드 URL 정리

**결론부터**: 개별 영상을 URL 하나로 받는 건 **안 됩니다.**
전부 수 GB짜리 `tar.gz` 묶음으로 올라가 있어요. 딱 하나 예외가 공식 데모 영상입니다.

---

## 1. 바로 받을 수 있는 것 — 공식 데모 영상 (2.3 MB)

```
https://huggingface.co/datasets/JungleGym/TimeLens-Assets/resolve/main/2Y8XQ.mp4
```

```bash
curl -fL -o 2Y8XQ.mp4 \
  "https://huggingface.co/datasets/JungleGym/TimeLens-Assets/resolve/main/2Y8XQ.mp4"
```

TimeLens 공식 모델 카드가 쓰는 그 영상입니다. 질의는 `A man drinks water with a glass`.

---

## 2. 평가 데이터 — TimeLens-Bench (질의 + 정답 있음) ★권장

레포: <https://huggingface.co/datasets/TencentARC/TimeLens-Bench>

### 주석 파일 — 작아서 바로 받을 수 있습니다

어떤 영상에 어떤 질의·정답이 있는지 먼저 볼 수 있어요.

```
https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/charades-timelens.json
https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/activitynet-timelens.json
https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/qvhighlights-timelens.json
```

형식:

```json
{
  "3MSZA": {
    "duration": 31.0,
    "spans":   [[25, 30], [1, 24], [0, 1], [23, 24]],
    "queries": ["A woman is repeatedly flipping the switch on the wall.", ...]
  }
}
```

키가 영상 ID, `spans[i]` 가 `queries[i]` 의 정답 구간입니다.

### 영상 — 샤드 단위 (수 GB)

| 데이터셋 | 평균 길이 | 샤드 | 개당 | 전체 |
|---|---|---|---|---|
| **charades** | 29.6초 | 1개 | 6.54 GB | **6.5 GB** ← 가장 가벼움 |
| activitynet | 134.9초 (최대 4분대) | 6개 | 8.4 GB | 50.2 GB |
| qvhighlights | 149.6초 | — | ~8 GB | — |

```
https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/video_shards/charades/charades_shard_01.tar.gz
https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/video_shards/activitynet/activitynet_shard_01.tar.gz
   ... activitynet_shard_02.tar.gz  ~  _06.tar.gz
```

---

## 3. 학습 데이터 — TimeLens-100K (**146 GB**, 비현실적)

레포: <https://huggingface.co/datasets/TencentARC/TimeLens-100K>

```
https://huggingface.co/datasets/TencentARC/TimeLens-100K/resolve/main/timelens-100k.jsonl   (12.6 MB)
https://huggingface.co/datasets/TencentARC/TimeLens-100K/tree/main/video_shards             (146 GB)
```

**전체가 146 GB 입니다.** 스토리지 128 GiB 에 안 들어갑니다.
학습을 직접 할 게 아니면 받을 이유가 없어요. 주석 파일(12.6MB)만 구경하는 정도로 충분합니다.

영상 출처: CosMo-Cap, InternVid-VTime, DiDeMo, QuerYD, HiREST

---

## 4. 가장 쉬운 방법 — 명령 한 줄 ★

제가 주석 파일을 미리 훑어서 **질의가 명확한 영상을 골라뒀습니다.**

```bash
python3 run_experiments.py --download-bench \
    --bench-pick "E6DLK,KOVTR,F7TG5,3MSZA" --cleanup-shard
```

이 한 줄이 하는 일:

1. 주석 파일 다운로드
2. 샤드 다운로드 (6.5GB)
3. **고른 4개 영상만** 압축 해제
4. 샤드 삭제 (용량 회수)
5. `videos/E6DLK.mp4` 처럼 바로 쓸 수 있게 배치
6. **각 영상의 질의와 정답 시각을 화면에 출력**

출력이 이렇게 나옵니다.

```
  ■ videos/E6DLK.mp4    charades-TimeLens / E6DLK (27초)
    ┌ 웹 UI 에 올린 뒤 아래 문장을 그대로 넣어보세요 ─────────────
    │  The man runs to the door and looks outside.
    │      -> 정답 11초 ~ 25초
    │  A man is sitting on a chair.
    │      -> 정답 5초 ~ 10초
    ...
```

그대로 웹 UI 에 복사해서 넣고, 나온 구간이 정답과 얼마나 맞는지 보면 됩니다.

### 추천 영상 (제가 골라둔 것)

| ID | 길이 | 질의 수 | 내용 |
|---|---|---|---|
| **E6DLK** | 27초 | **6개** | 의자에 앉았다 일어나 문으로 뛰어감. 동작이 많아 데모에 좋음 |
| **KOVTR** | 35초 | 3개 | 소파에서 일어남 → 슬리퍼 신음 → 약 먹음. 순서가 뚜렷함 |
| **F7TG5** | 21초 | 4개 | 코트 벗음, 의자에 앉음, 담요 덮음 |
| **3MSZA** | 31초 | 4개 | 스위치 켜고 끔, 과자 먹음, 문틀에 기댐 |
| VXJS4 | 30초 | 2개 | 문 열고 통과 (짧고 단순) |
| AKO6M | 19초 | 2개 | 욕실에서 컵 들고 서 있음, 수납장에서 가방 꺼냄 |

**E6DLK 를 첫 데모로 추천합니다.** 질의가 6개라 한 영상으로 여러 번 보여줄 수 있고,
`The man gets up from the chair`(10~11초) vs `The man sits down on the chair`(4~6초)
처럼 **비슷하지만 반대인 동작**이 있어서 모델이 문장을 읽는지 확인하기 좋습니다.

---

## 5. 수동으로 하고 싶다면

샤드를 통째로 받은 뒤 **원하는 영상만 꺼내고 샤드는 지웁니다.**

```bash
# ① 주석부터 (작음) — 어떤 영상이 있는지 확인
curl -fL -o charades-timelens.json \
  "https://huggingface.co/datasets/TencentARC/TimeLens-Bench/resolve/main/charades-timelens.json"

# ② 마음에 드는 영상 ID 고르기 (질의가 명확한 것으로)
python3 -c "
import json
a = json.load(open('charades-timelens.json'))
for vid, r in list(a.items())[:10]:
    print(f\"{vid}  {r['duration']:.0f}초\")
    for q, s in zip(r['queries'], r['spans']):
        print(f'    {s}  {q}')
"

# ③ 샤드 받기 (6.5GB, 인스턴스에서 하면 몇 분)
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download TencentARC/TimeLens-Bench \
  --repo-type=dataset \
  --include "video_shards/charades/charades_shard_01.tar.gz" \
  --local-dir bench_dl

# ④ 원하는 영상만 추출
mkdir -p picked
tar -xzf bench_dl/video_shards/charades/charades_shard_01.tar.gz \
    -C picked --wildcards '*3MSZA*' '*AMT7R*'

# ⑤ 샤드 삭제 (용량 회수)
rm -rf bench_dl
ls picked/
```

`--wildcards` 에 원하는 ID를 나열하면 그것만 꺼냅니다.
압축 해제 자체는 파일 전체를 훑지만, **디스크에 쓰이는 건 고른 것뿐**입니다.

---

## 6. 그래서 어떻게 쓰나

**네, 받은 영상을 웹 UI에 올리면 됩니다.**

```
① 영상 확보 (위 방법 중 하나)
② python3 app.py 실행
③ 브라우저에서 영상 끌어다 놓기
④ 질의 입력 → 구간 + 클립 확인
```

> **중요** — 다운로드는 **엘리스 인스턴스에서** 하세요.
> 맥북에 6.5GB를 받아서 다시 업로드하면 가정용 회선 업로드 속도 때문에
> 훨씬 오래 걸립니다. 인스턴스는 데이터센터 회선이라 몇 분이면 끝나요.

**정량 평가(tIoU)까지 필요하면** 수동으로 받을 필요 없습니다.
아래 한 줄이 다운로드·추출·질의·정답 등록을 전부 합니다.

```bash
python3 run_experiments.py --download-bench --bench-dataset charades
python3 run_experiments.py --list-videos
python3 run_experiments.py --model safe
```

---

## 7. 참고 — 원본 데이터셋 (TimeLens 정제 전)

정제 전 원본이 필요하다면 (보통 필요 없습니다):

| 데이터셋 | 링크 |
|---|---|
| Charades-STA | <https://github.com/jiyanggao/TALL> |
| ActivityNet Captions | <https://cs.stanford.edu/people/ranjaykrishna/densevid/> |
| QVHighlights | <https://github.com/jayleicn/moment_detr> |

Charades 원본 영상은 별도 등록·신청이 필요하고, ActivityNet 은 YouTube 링크라
영상이 내려가 있는 경우가 많습니다. **TimeLens-Bench 쪽이 훨씬 편합니다.**

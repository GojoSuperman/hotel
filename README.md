# hotel — 호텔 룸 3D 재구성

호텔 방 사진으로 3D 가우시안 스플래팅(3DGS)을 만드는 구글 코랩 노트북 모음.

## 노트북

| 파일 | 용도 | 입력 |
|---|---|---|
| `호텔룸_InstantSplat_colab.ipynb` | **사진 몇 장(2~12장)으로 빠른 3DGS** — COLMAP 생략, 검증 완료(2026-08-10) | 드라이브 `instantsplat_input/`에 JPG/PNG 2~12장 |
| `호텔룸_3DGS_colab.ipynb` | 표준 3DGS(nerfstudio Splatfacto) — 고품질, 상업 이용에 유리 | 드라이브 `gsplat_input/`에 사진 50장+ 또는 영상 |
| `파노라마_3D_colab.ipynb` | **360도 사진 1장 → 3D** — AI 깊이 추정(Depth Anything V2, Apache-2.0) 기반 | 드라이브 `pano_input/`에 2:1 등장방형 사진 1장 |

노트북으로 여는 방법: [colab.research.google.com](https://colab.research.google.com) → 파일 → 노트 열기 → GitHub 탭 → `GojoSuperman/hotel` 검색 → 노트북 클릭.

노트북 대신 **아래 가이드를 셀 단위로 복붙**해서 새 코랩 노트북에서 실행해도 똑같이 동작한다.

---

# InstantSplat 단계별 실행 가이드 (복붙용)

## 사전 준비 (코랩 밖에서)

1. 구글 드라이브 최상위에 `instantsplat_input` 폴더 생성 (정확히 이 이름, 소문자+언더바)
2. 호텔 방 사진 **2~12장** 업로드 — **JPG/PNG만** 인식 (아이폰 HEIC는 JPG로 변환)
   - 같은 방을 서로 다른 위치에서, 인접 사진끼리 시야가 절반 이상 겹치게
3. 코랩에서 새 노트북 생성 → 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** 선택

> ⚠️ 무료 코랩은 자리를 오래 비우면 가상 컴퓨터가 초기화된다. 셀 1~8을 **한 번에 쭉** 실행할 것.
> 초기화되면 (사진은 드라이브에 남아 있으니) 셀 1부터 다시 실행하면 된다.

## 셀 1 — GPU 확인 (몇 초)

```python
!nvidia-smi
```

✅ 출력 표에 `Tesla T4`가 보이면 성공. 안 보이면 런타임 유형을 다시 확인.

## 셀 2 — 레포 클론 + 모델 다운로드 (3~5분)

```python
%cd /content
!rm -rf /content/InstantSplat
!git clone --recursive https://github.com/NVlabs/InstantSplat.git
%cd /content/InstantSplat
!mkdir -p mast3r/checkpoints/
!wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P mast3r/checkpoints/
```

✅ `Submodule ... checked out` 여러 줄 + 다운로드 진행 막대가 `100%`(2.57G)까지 가면 성공.

## 셀 3 — 설치 + PyTorch 호환 패치 (약 10분, 가장 오래 걸림)

```python
%cd /content/InstantSplat
!pip install -q -r requirements.txt
!pip install -q ./submodules/simple-knn
!pip install -q ./submodules/diff-gaussian-rasterization
!pip install -q ./submodules/fused-ssim
!grep -rl "map_location='cpu')" /content/InstantSplat/mast3r /content/InstantSplat/dust3r --include=*.py | xargs -r sed -i "s/map_location='cpu')/map_location='cpu', weights_only=False)/g"
print("설치 + 패치 완료")
```

- 노란/빨간 **경고(warning)가 잔뜩 나와도 정상** (CUDA 컴파일 특성). 출력이 몇 분간 조용해도 중단 금지.
- 마지막 `grep|sed` 줄은 필수 패치다. 코랩의 PyTorch 2.6+에서 체크포인트 로드가 `UnpicklingError`로 죽는 것을 막는다.

✅ 마지막에 `설치 + 패치 완료`가 출력되면 성공. `ERROR:`로 멈추면 런타임 → 세션 다시 시작 후 셀 2부터.

## 셀 4 — 드라이브 연결 + 사진 가져오기 (1분)

```python
SCENE = 'room1'  # ← 새 방/새 사진 세트마다 여기만 바꾸기 (영문/숫자)

from google.colab import drive
drive.mount('/content/drive')

import os, shutil, glob
INPUT_DIR = '/content/drive/MyDrive/instantsplat_input'
assert os.path.isdir(INPUT_DIR), 'instantsplat_input 폴더가 드라이브에 없습니다. 만들고 사진을 넣어주세요.'

SOURCE_PATH = f'/content/InstantSplat/assets/hotel/{SCENE}'
shutil.rmtree(SOURCE_PATH, ignore_errors=True)  # 이전 실행 잔여물 제거
os.makedirs(f'{SOURCE_PATH}/images')

imgs = sorted(glob.glob(f'{INPUT_DIR}/*.[jJpP][pPnN]*[gG]'))
assert 2 <= len(imgs) <= 12, f'사진이 {len(imgs)}장입니다. 2~12장(JPG/PNG)을 넣어주세요.'
for i, p in enumerate(imgs):
    shutil.copy(p, f'{SOURCE_PATH}/images/{i:03d}{os.path.splitext(p)[1].lower()}')

N_VIEWS = len(imgs)
MODEL_PATH = f'/content/InstantSplat/output_infer/hotel/{SCENE}/{N_VIEWS}_views'
GS_ITER = 1000  # 품질을 올리려면 2000~3000
print(f'[{SCENE}] 사진 {N_VIEWS}장 준비 완료')
```

- 실행하면 구글 계정 접근 허용 팝업이 뜬다 → 계정 선택 후 "허용".

✅ `[room1] 사진 N장 준비 완료`가 나오면 성공.
- `폴더가 드라이브에 없습니다` → 폴더 이름 오타이거나 내 드라이브 최상위가 아님
- `사진이 0장입니다` → HEIC 등 미지원 형식. JPG로 변환 후 이 셀만 재실행

## 셀 5 — 기하 초기화 (1~3분)

MASt3R 모델이 사진들만 보고 각 사진의 카메라 위치와 방의 3D 포인트를 추정한다 (COLMAP 대체 — InstantSplat의 핵심).

```python
%cd /content/InstantSplat
!python ./init_geo.py -s {SOURCE_PATH} -m {MODEL_PATH} \
  --n_views {N_VIEWS} --focal_avg --co_vis_dsp --conf_aware_ranking --infer_video
```

- `cannot find cuda-compiled version of RoPE2D, using a slow pytorch version` 경고는 무시 (조금 느릴 뿐 결과 동일).

✅ `Traceback` 없이 끝나면 성공.

## 셀 6 — 학습 (1~2분)

```python
!python ./train.py -s {SOURCE_PATH} -m {MODEL_PATH} -r 1 \
  --n_views {N_VIEWS} --iterations {GS_ITER} --pp_optimizer --optim_pose
```

✅ 진행이 `1000/1000`(GS_ITER 값)까지 가면 성공.
- `confidence_dsp.npy` 없음 오류 → 셀 5가 중간에 죽은 것. 셀 5 출력의 Traceback을 확인

## 셀 7 — 렌더링 (1~2분)

```python
!python ./render.py -s {SOURCE_PATH} -m {MODEL_PATH} -r 1 \
  --n_views {N_VIEWS} --iterations {GS_ITER} --infer_video
```

## 셀 8 — 결과를 드라이브에 저장 (몇 초)

```python
import glob, shutil

plys = glob.glob(f'{MODEL_PATH}/**/point_cloud.ply', recursive=True)
assert plys, 'point_cloud.ply를 찾지 못했습니다. 5~7단계 출력의 오류를 확인하세요.'
shutil.copy(sorted(plys)[-1], f'/content/drive/MyDrive/instantsplat_{SCENE}.ply')
print(f'저장: MyDrive/instantsplat_{SCENE}.ply')

for i, v in enumerate(glob.glob(f'{MODEL_PATH}/**/*.mp4', recursive=True)):
    shutil.copy(v, f'/content/drive/MyDrive/instantsplat_{SCENE}_video_{i}.mp4')
    print(f'저장: MyDrive/instantsplat_{SCENE}_video_{i}.mp4')
```

✅ `저장: ...` 두 줄이 나오면 전 과정 완료.

## 결과 보기

1. **영상**: 드라이브에서 `instantsplat_<장면>_video_0.mp4` 더블클릭 → 사진 위치 사이를 이동하는 카메라 영상
2. **3D 조작**: `instantsplat_<장면>.ply` 다운로드 → [superspl.at/editor](https://superspl.at/editor)에 드래그
   - 왼쪽 드래그 = 회전, 휠 = 확대/축소, 오른쪽 드래그 = 이동
   - 떠다니는 잡티는 박스 선택 → Delete로 제거, `.splat` 압축 변환도 가능

## 다른 사진으로 반복 작업

1. 드라이브 `instantsplat_input`의 기존 사진을 **지우고** 새 사진으로 교체
2. 셀 4의 `SCENE` 이름 변경 (예: `room2`) → 셀 4~8만 재실행 (세션이 살아있으면 셀 1~3 생략)

## 내 HTML에서 보기 (`viewer.html`)

superspl.at 대신 자체 뷰어로 볼 수 있다 ([GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) 라이브러리 사용, 인터넷 연결 필요).

**가장 쉬운 방법 (설치·서버 불필요):**

1. **https://gojosuperman.github.io/hotel/viewer.html** 접속
2. 드라이브에서 내려받은 `instantsplat_….ply` 파일을 화면에 **드래그** (또는 클릭해서 선택)

파일은 브라우저 안에서만 열리고 어디에도 업로드되지 않는다. 다른 파일을 보려면 새로고침 후 다시 드래그.

팁:
- **파일 용량 줄이기**: 원본 .ply는 수십~수백 MB일 수 있다. [superspl.at/editor](https://superspl.at/editor)에서 잡티 제거 후 `.splat`으로 내보내면 크게 줄어든다. 뷰어는 .ply/.splat/.ksplat 모두 지원.
- **장면이 뒤집혀 보이면**: `viewer.html`의 `cameraUp: [0, -1, 0]`을 `[0, 1, 0]`으로 수정.
- **드래그 없이 자동 로드**(사이트 임베드용): splat 파일을 저장소에 커밋하고 `viewer.html?file=scene.ply`처럼 주소로 지정하면 접속 즉시 로드된다 (GitHub 파일당 100MB 제한 유의). 로컬 개발 시엔 같은 폴더에서 `python3 -m http.server 8000` 후 `http://localhost:8000/viewer.html?file=scene.ply`.

## 알아둘 것

- **품질 기대치**: 사진이 찍힌 위치 근처 시점만 온전하다. 사진에 없는 각도는 구멍/늘어짐이 정상. 거울·유리·순백 벽면은 3DGS 공통 약점.
- **라이선스**: InstantSplat은 NVIDIA 연구용 라이선스 — 상업 서비스 사용 전 [레포의 LICENSE](https://github.com/NVlabs/InstantSplat) 확인. 상업 목적이면 표준 3DGS 노트북(`호텔룸_3DGS_colab.ipynb`) 경로가 안전.
- **`CUDA out of memory`**: 사진 장수를 줄이거나(6~8장) 긴 변 1600px 이하로 리사이즈.
- **`init_geo.py: No such file or directory`**: 세션이 초기화된 것. 셀 2부터 다시.

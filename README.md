# hotel — 호텔 룸 3D 재구성

호텔 방 사진으로 3D 가우시안 스플래팅(3DGS)을 만드는 구글 코랩 노트북 모음.

## 노트북

| 파일 | 용도 | 입력 |
|---|---|---|
| `호텔룸_InstantSplat_colab.ipynb` | **사진 몇 장(2~12장)으로 빠른 3DGS** — COLMAP 생략, 검증 완료(2026-08-10) | 드라이브 `instantsplat_input/`에 JPG/PNG 2~12장 |
| `호텔룸_3DGS_colab.ipynb` | 표준 3DGS(nerfstudio Splatfacto) — 고품질, 상업 이용에 유리 | 드라이브 `gsplat_input/`에 사진 50장+ 또는 영상 |

## 사용법

1. [colab.research.google.com](https://colab.research.google.com) → 파일 → 노트 열기 → **GitHub 탭**에서 이 저장소의 노트북을 바로 열기 (또는 파일 업로드)
2. 런타임 유형을 **T4 GPU**로 변경
3. 노트북 상단 안내대로 드라이브에 입력 폴더 준비 후 셀을 순서대로 실행
4. 결과(.ply)는 [SuperSplat 편집기](https://superspl.at/editor)에 드래그해서 확인

## 참고

- 무료 코랩은 유휴 시 초기화됨 → 노트북을 한 번에 쭉 실행할 것
- InstantSplat 결과물은 NVIDIA 연구용 라이선스 — 상업 서비스 사용 전 확인 필요

# gemma4 로컬 서빙

`gemma-4-12B-it` 멀티모달 모델(12B, 4bit)을 로컬 OpenAI 호환 서버로 구동한다. 백엔드는 OS로 런타임 자동 선택된다:

- **macOS (Apple Silicon)** — [MLX](https://github.com/ml-explore/mlx) `mlx_vlm.server` (`:8080`)
- **WSL / Linux (NVIDIA CUDA)** — [Ollama](https://ollama.com) (`:11434`)

`webapp.py`·`chat.py`·`templates/`(UI)는 두 백엔드가 공유한다. `GEMMA_BACKEND=mlx|ollama` 로 강제할 수 있다.

아래 **설치·서버** 절은 macOS(MLX) 기준이다. WSL/CUDA는 [WSL / CUDA (Ollama)](#wsl--cuda-ollama) 절을 참고한다.

## 환경

- macOS (Apple Silicon) · Python 3.12 — MLX 백엔드
- WSL / Linux + NVIDIA GPU — Ollama 백엔드 ([WSL / CUDA (Ollama)](#wsl--cuda-ollama))
- 모델: [`mlx-community/gemma-4-12B-it-4bit`](https://huggingface.co/mlx-community/gemma-4-12B-it-4bit) (~7GB, 추론 시 메모리 ~11GB) / Linux는 `gemma4:12b`

## 설치

```bash
cd ~/workspace/github.com/nalbam/gemma4

# 패키지
pip install -r requirements.txt

# 모델 다운로드 (~7GB)
hf download mlx-community/gemma-4-12B-it-4bit

# 8bit도 쓰려면 함께 받는다 (~12.7GB)
hf download mlx-community/gemma-4-12B-it-8bit
```

## 서버

### 스크립트 (권장)

```bash
./scripts/start.sh         # 4bit로 백그라운드 기동 + 준비 완료까지 대기 (기본)
./scripts/start.sh 8bit    # 8bit로 기동
./scripts/stop.sh          # 중지
```

### 띄우기 (직접)

```bash
# 포그라운드 (Ctrl-C로 중지)
python -m mlx_vlm.server \
  --model mlx-community/gemma-4-12B-it-4bit \
  --host 127.0.0.1 --port 8080

# 백그라운드 (터미널 닫아도 유지)
nohup python -m mlx_vlm.server \
  --model mlx-community/gemma-4-12B-it-4bit \
  --host 127.0.0.1 --port 8080 > /tmp/gemma4_server.log 2>&1 &
```

아래 로그가 뜨면 준비 완료:

```
Model ready, continuous batching enabled.
Uvicorn running on http://127.0.0.1:8080
```

엔드포인트: `http://127.0.0.1:8080/v1`

### 상태 확인

```bash
pgrep -fl mlx_vlm.server                                              # 프로세스
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/v1/models  # 응답(200=정상)
tail -f /tmp/gemma4_server.log                                        # 로그
./scripts/monitor.sh        # 막대 그래프 (간격(초) 인자, 기본 2)
```

**전체 시스템**
- **CPU** — `top`의 user+sys (전체 코어 기준, 0~100%).
- **MEM** — 시스템 물리 메모리 사용량(`top` PhysMem used). macOS가 캐시·압축을 계속 조절해 미세 변동한다.

**gemma4 프로세스**
- **CPU** — raw % (1코어=100% 기준). 추론은 GPU 위주라 CPU는 보조적으로만 오른다.
- **MEM** — `footprint`(phys_footprint)로 모델 **실제 점유량** 표시(`ps` RSS는 MLX/Metal 메모리를 못 잡아 부정확). 추론해도 거의 안 변하는 게 정상 — 모델은 **로드 시점에 메모리를 점유**하고, 추론은 GPU 연산이라 메모리 증감이 없다(KV cache만 미세 증가).

MLX 추론의 실제 부하(GPU)를 보려면 별도로(sudo 필요):

```bash
sudo powermetrics --samplers gpu_power -i 1000 -n 1
```

### 중지

```bash
pkill -f mlx_vlm.server   # 백그라운드
# 포그라운드는 해당 터미널에서 Ctrl-C
```

## 사용법

### chat.py — 텍스트 대화 (스트리밍)

```bash
# 단발
python chat.py "사과에 대해 설명해줘"

# 대화형 (멀티턴, 종료: exit / quit / Ctrl-D)
python chat.py
```

### curl

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/gemma-4-12B-it-4bit","messages":[{"role":"user","content":"ping"}],"max_tokens":50}'
```

### OpenAI SDK

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")  # 키는 아무 값
r = client.chat.completions.create(
    model="mlx-community/gemma-4-12B-it-4bit",
    messages=[{"role": "user", "content": "안녕"}],
)
print(r.choices[0].message.content)
```

## 웹 UI

서버 제어·자원 모니터링·채팅을 한 화면에서 다루는 관리 콘솔.

```bash
python webapp.py   # http://127.0.0.1:8000
```

의존성은 위 `requirements.txt`에 포함되어 별도 설치는 필요 없다.

- **서버 제어** — 4bit/8bit를 드롭다운에서 골라 시작/중지한다 (`start.sh`/`stop.sh` 호출). 실행 중 다른 정밀도를 고르고 시작하면 자동으로 재기동해 전환한다. 서버가 꺼져 있어도 UI에서 바로 기동할 수 있다.
- **모니터링** — 전체 시스템과 gemma4 프로세스의 CPU·MEM을 2초 간격으로 갱신한다 (`monitor.sh`와 같은 지표).
- **채팅** — 스트리밍 응답, 마크다운·표 렌더링, 이미지 첨부(멀티모달)를 지원한다. 서버가 꺼져 있으면 먼저 시작하라는 안내가 뜬다.

## WSL / CUDA (Ollama)

WSL(Ubuntu) + NVIDIA GPU 환경. RTX 4070 SUPER(12GB) 기준 **4bit(Q4_K_M, ~7GB)는 전부 VRAM에 적재**된다.

### 준비

```bash
# Ollama 0.20+ 필요 (Gemma 4 지원). 구버전이면 먼저 업데이트 (sudo):
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드 + Python 의존성
./scripts/setup-wsl.sh
```

`scripts/setup-wsl.sh`는 ollama 버전을 확인하고 `gemma4:12b`를 받은 뒤 `requirements-wsl.txt`(mlx 제외)를 설치한다.

### 서버

Ollama 데몬(systemd)이 서빙을 담당한다. 모델은 첫 요청 또는 웹 UI **"시작"**으로 VRAM에 적재된다.

```bash
ollama ps                  # 로드된 모델 확인
ollama run gemma4:12b      # 터미널에서 바로 대화
```

엔드포인트(OpenAI 호환): `http://127.0.0.1:11434/v1`

### 사용법

`chat.py`·`webapp.py`는 Linux에서 자동으로 `:11434`를 향한다 (`GEMMA_BASE_URL`로 강제 가능).

```bash
python chat.py "사과에 대해 설명해줘"   # 단발
python chat.py                          # 대화형
python webapp.py                        # 웹 UI → http://127.0.0.1:8000
```

- **모니터링** — 웹 UI와 `scripts/monitor.sh`의 "gemma4" 패널은 Linux에서 **GPU(util·VRAM)**를 표시한다 (`nvidia-smi`).
- **8bit** — `gemma4:12b-q8_0`(~13GB)는 12GB VRAM을 초과해 CPU 오프로딩으로 느려진다. 4bit 권장.
- **이미지** — 웹 UI의 📎 첨부로 멀티모달(vision) 질의가 가능하다 (Gemma 4 멀티모달).

## 참고 (macOS / MLX)

- 응답에 `<audio|>`·`<image|>` 같은 멀티모달 특수 토큰이 가끔 섞여 나올 수 있다 (서버 디코딩 아티팩트). 필요하면 클라이언트에서 후처리로 제거한다.
- 4bit/8bit는 웹 UI 드롭다운 또는 `./scripts/start.sh 8bit`로 전환한다. 8bit(~12.7GB)가 품질이 더 높고, 4bit(~11GB) 대비 메모리 차이는 작다.
- 슬립 방지(장시간 서빙): `sudo pmset -a sleep 0`

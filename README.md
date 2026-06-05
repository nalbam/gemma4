# gemma4 로컬 서빙 (MLX)

Apple Silicon Mac에서 [MLX](https://github.com/ml-explore/mlx)로 `gemma-4-12B-it-4bit` 멀티모달 모델을 로컬 OpenAI 호환 서버로 구동한다.

## 환경

- macOS (Apple Silicon)
- Python 3.12
- 모델: [`mlx-community/gemma-4-12B-it-4bit`](https://huggingface.co/mlx-community/gemma-4-12B-it-4bit) (~7GB, 추론 시 메모리 ~11GB)

## 설치

```bash
cd ~/workspace/github.com/nalbam/gemma4

# 패키지
pip install -r requirements.txt

# 모델 다운로드 (~7GB)
hf download mlx-community/gemma-4-12B-it-4bit
```

## 서버

### 스크립트 (권장)

```bash
./start.sh   # 백그라운드 기동 + 준비 완료까지 대기
./stop.sh    # 중지
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
./monitor.sh        # 막대 그래프 (간격(초) 인자, 기본 2)
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

- **서버 제어** — 시작/중지 버튼이 `start.sh`/`stop.sh`를 호출한다. gemma 서버가 꺼져 있어도 UI에서 바로 기동할 수 있다.
- **모니터링** — 전체 시스템과 gemma4 프로세스의 CPU·MEM을 2초 간격으로 갱신한다 (`monitor.sh`와 같은 지표).
- **채팅** — 스트리밍 응답, 마크다운·표 렌더링, 이미지 첨부(멀티모달)를 지원한다. 서버가 꺼져 있으면 먼저 시작하라는 안내가 뜬다.

## 참고

- 응답에 `<audio|>`·`<image|>` 같은 멀티모달 특수 토큰이 가끔 섞여 나올 수 있다 (서버 디코딩 아티팩트). 필요하면 클라이언트에서 후처리로 제거한다.
- 더 높은 품질이 필요하면 8bit(`mlx-community/gemma-4-12B-it-8bit`, ~12.7GB)로 교체할 수 있다. 메모리 여유를 확인한다.
- 슬립 방지(장시간 서빙): `sudo pmset -a sleep 0`

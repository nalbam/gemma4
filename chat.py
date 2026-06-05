"""gemma 서버에 메시지를 보내고 응답을 출력한다.

사용법:
    python chat.py "안녕, 뭐 할 수 있어?"   # 한 번 보내고 응답 출력
    python chat.py                          # 대화형 모드 (멀티턴, exit/quit 로 종료)
"""
import json
import sys
import urllib.request

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "mlx-community/gemma-4-12B-it-4bit"


def send(messages: list[dict], max_tokens: int = 512) -> str:
    """토큰을 받는 즉시 출력하고, 전체 응답 텍스트를 반환한다."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    parts: list[str] = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0]["delta"].get("content")
            if delta:
                print(delta, end="", flush=True)
                parts.append(delta)
    print()
    return "".join(parts)


def once(prompt: str) -> None:
    send([{"role": "user", "content": prompt}])


def interactive() -> None:
    print("대화형 모드 (종료: exit / quit / Ctrl-D)")
    messages: list[dict] = []
    while True:
        try:
            user = input("\n나> ").strip()
        except EOFError:
            print()
            break
        if user in ("exit", "quit"):
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        print("\ngemma> ", end="", flush=True)
        reply = send(messages)
        messages.append({"role": "assistant", "content": reply})


def main() -> None:
    if len(sys.argv) > 1:
        once(" ".join(sys.argv[1:]))
    else:
        interactive()


if __name__ == "__main__":
    main()

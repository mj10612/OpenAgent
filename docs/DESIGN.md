# OpenAgent 설계 문서

> 목표: **어떤 모델이든** 쓸 수 있는 터미널 기반 AI Coding Agent.
> 라이선스: Apache-2.0

## 1. 핵심 원칙

### 1.1 "모든 모델 지원"을 가능하게 하는 3중 안전망

모델 지원의 본질적인 문제는 *프로토콜*이지 *모델 목록*이 아니다. 세계의 모든 LLM은 결국
소수의 프로토콜로 수렴한다. OpenAgent는 계층별로 세 번의 탈출구를 둔다.

| 계층 | 방식 | 새로 지원해야 하는 것 | 커버리지 |
| --- | --- | --- | --- |
| L1 | **네이티브 어댑터** | 어댑터 1개 | Anthropic Messages, Google Gemini, Azure OpenAI, AWS Bedrock, Ollama |
| L2 | **OpenAI 호환 어댑터** | 설정만 | OpenAI, DeepSeek, Groq, xAI, Mistral, Together, Fireworks, OpenRouter, Cerebras, sambanova, vLLM, LM Studio, llama.cpp, NVIDIA NIM, Sakana… (~40곳, 계속 증가) |
| L3 | **범용 스키마 매핑** | 설정만 | 위 어디에도 해당하지 않는 **임의의 HTTP JSON API**. 환경변수 경로 템플릿 + JSONPath |

> 새 프로토콜이 등장하면 (1) 어댑터를 추가하거나, (2) `custom` 어댑터에 JSONPath 매핑만
> 적는다. **코드를 한 줄도 바꾸지 않고** 새 API가 지원된다.

### 1.2 도구 호출(Tool Calling) 3단 지원

가장 많은 모델이 "약한" 지점이 도구 호출이다. 4단으로 격리한다.

- **T0 — 네이티브**: 프로바이더가 `tools` 필드를 그대로 렌더링하고 tool_call을 파싱한다.
- **T1 — OpenAI 호환**: `tools` + `tool_calls` JSON 스키마. 요청/응답 필드만 재배치.
- **T2 — 텍스트 프로토콜**: native tool을 지원하지 않는 모델용. 시스템 프롬프트에
  도구 사양을 서술하고, 모델이 ```tool_code 블록을 내면 파서가 실행한다.
- **T3 — 없음**: 도구 없이 순수 프롬프트 모드. 최후의 폴백.

에이전트 루프는 도구 호출 방식을 **몰라야 하고**, 프로바이더의 `tool_protocol` 속성만
보고 분기한다. 새로운 도구 호출 문법이 나오면 프로바이더 하나만 바꾸면 된다.

### 1.3 컨텍스트 관리

긴 세션에서 무한 컨텍스트는 불가능하므로:

- **토큰 예산**: 프로바이더별 컨텍스트 윈도우 + 예약(reasoning) 분량을 고려해 예산을 계산.
- **자동 압축(compaction)**: 임계치를 넘으면 LLM이 대화 요약을 생성 → 요약 메시지 1개로 교체.
- **Sliding window**: 압축 후에도 넘치면 가장 오래된 완전한 턴 단위로 드롭.
- **경계 정합성**: 턴 경계를 절대 자르지 않는다. 도구 요청/응답 쌍을 쪼개지 않는다.

### 1.4 안전성

- **권한(permission)**: 도구별 allow/ask/deny 3상태. `fs_write`, `shell`는 기본 `ask`.
- **경로 샌드박스**: 작업 루트를 벗어나는 쓰기/읽기를 차단(심볼릭 링크 해석 후 검증).
- **Shell 인젝션 방지**: 도구 인자는 shell로 감싸지 않고 배열로 전달.
- **프롬프트 인젝션 방어**: MCP/웹 콘텐츠는 신뢰 경계 밖 데이터로 마킹하고 시스템 프롬프트에 명시.
- **시크릿 마스킹**: 로그/트레이스에 API 키 노출 방지.

## 2. 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI  /  TUI                           │
│              (Rich 기반 REPL, 스트리밍 렌더링)                 │
└───────────────────────────┬──────────────────────────────────┘
                            │  사용자 입력 / 이벤트
┌───────────────────────────▼──────────────────────────────────┐
│                      AgentRunner                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────┐  │
│  │ Prompt     │ │  Context   │ │  Tool      │ │ Session   │  │
│  │ Builder    │ │ Manager    │ │  Registry  │ │ Store     │  │
│  │            │ │ (토큰예산/ │ │            │ │           │  │
│  │ 시스템프롬 │ │  압축/윈도 │ │ builtin+MCP│ │ (JSONL    │  │
│  │프트 조립   │ │  우 처리)  │ │            │ │  세션저장)│  │
│  └────────────┘ └────────────┘ └────────────┘ └───────────┘  │
└──────┬───────────────────────────────┬───────────────────────┘
       │ CompletionRequest             │ ToolCall
┌──────▼───────────────────────────────▼───────────────────────┐
│                   ProviderRouter / Registry                   │
│  라우팅: model-ref → provider preset → base_url/api_key/params │
│  지원 범위: native / openai-compatible / custom(JSONPath)     │
└──────┬───────────────────────────────────────────────────────┘
       │  단일 chat completions 또는 여러 프로토콜로 분기
┌──────▼───────────────────────────────────────────────────────┐
│                  Transport (httpx)  +  SSE 파서              │
└──────────────────────────────────────────────────────────────┘
                            │
                      임의의 LLM API
```

### 2.1 모듈 레이아웃

```
src/openagent/
├── core/
│   ├── types.py        # Message, ToolCall, ToolResult, ChatRequest …
│   ├── events.py       # 스트리밍 이벤트 유니온
│   ├── provider.py     # ChatProvider ABC + ProviderRegistry
│   ├── presets.py      # 내장 프로바이더 프리셋 (40+)
│   └── router.py       # 모델 레퍼런스 파싱 및 라우팅
├── context/
│   ├── estimator.py    # 토큰 추정
│   ├── messages.py     # 누적/트리밍/경계 정합성
│   └── compactor.py    # LLM 요약 압축
├── prompts/
│   ├── base.py         # 시스템 프롬프트 빌더
│   └── assets/         # 시스템 프롬프트 텍스트
├── tools/
│   ├── base.py         # Tool ABC
│   ├── registry.py     # 등록 및 실행
│   ├── fs.py           # read/write/edit/glob/grep/ls
│   ├── shell.py        # 실행 (배열 전달)
│   ├── agent_tools.py  # todo, think, webfetch, websearch …
│   └── mcp/
│       ├── client.py   # stdio JSON-RPC
│       ├── http.py     # streamable HTTP / SSE
│       └── manager.py  # 다중 서버 수명주기
├── providers/
│   ├── openai_compat.py
│   ├── anthropic.py
│   ├── gemini.py
│   ├── azure.py
│   ├── ollama.py
│   ├── bedrock.py
│   └── custom.py       # JSONPath 범용 어댑터
├── session/
│   └── store.py        # JSONL 영속화, --resume
├── tui/
│   └── app.py
├── config.py
└── cli.py
```

## 3. 핵심 데이터 모델

### 3.1 메시지 (`core/types.py`)

```python
@dataclass(slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentPart]
    tool_calls: list[ToolCall]
    tool_call_id: str | None
    name: str | None
```

`content`는 멀티모달 대비 항상 리스트로 통일한다. 텍스트만 쓸 땐
`ContentPart(text=...)` 한 개. 이렇게 하면 비전 지원 모델로의 업그레이드가
전면 변경 없이 이루어진다.

### 3.2 스트리밍 이벤트 (`core/events.py`)

```python
type StreamEvent = StartEvent | TextDelta | ThinkingDelta | ToolCallStart \
                 | ToolCallDelta | ToolCallEnd | UsageEvent | DoneEvent | ErrorEvent
```

모든 프로바이더는 자기 네이티브 이벤트를 이 유니온으로 **정규화**한다. TUI와
에이전트 루프는 프로바이더를 전혀 모른다.

### 3.3 도구 결과 규약

- 성공: `content`에 텍스트, `is_error=False`
- 실패: 예외를 문자열로 변환해 모델에 전달하되 `is_error=True`.
  예외로 루프를 죽이지 않는다. 모델이 스스로 오류를 수정하고 재시도하게 한다.
- 결과 크기 상한: 200KB 초과 시 잘라내고 위치를 명시한다(컨텍스트 폭발 방지).

## 4. Provider 추상화

```python
class ChatProvider(ABC):
    name: str
    tool_protocol: ToolProtocol      # native | json_schema | text | none
    supports_streaming: bool

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]: ...
    async def list_models(self) -> list[ModelInfo]: ...
```

`ChatRequest`는 프로바이더 비종속. 어댑터가 `_render_messages()`로 자기 형식으로
변환한다. **어떤 모델이든 이 한 인터페이스만 구현하면 동작한다.**

## 5. 에이전트 루프

```
loop:
  1. 시스템 프롬프트 + 환경 컨텍스트 조립
  2. 토큰 예산 초과 시 → 압축 → 초과 시 → 슬라이딩 윈도우
  3. 프로바이더에 ChatRequest 전달, 스트림 수신
  4. 텍스트/생각은 TUI로 즉시 출력
  5. tool_calls가 있으면:
       a. 권한 검사 (allow/ask/deny)
       b. 병렬 실행 (asyncio.gather, 실패 격리)
       c. 결과를 tool role 메시지로 추가
       d. → 2로 복귀
  6. tool_calls 없으면 → 종료, usage 기록
```

**핵심 불변식**: `tool_calls`를 만들지 않은 assistant 메시지 뒤에 `tool` role
메시지가 오면 안 된다. 어댑터가 규약을 어기면 `Normalizer`가 자동 복구한다.

## 6. Provider 결정 흐름

```
사용자가 "gpt-5" 지정
  → Router가 알려진 프리셋과 매칭 (정규화: 소문자, 접두/접미 제거)
  → 없으면 "openai호환 + 기본 base_url"로 폴백 + 경고
  → 애초에 매칭 안 되면 사용자에게 등록向导 제시
```

## 7. 설정

`~/.config/openagent/config.toml` (또는 `OPENAGENT_CONFIG`):

```toml
[models.default]
provider = "openai-compat"
model     = "gpt-4o-mini"
base_url  = "https://api.openai.com/v1"
api_key   = "sk-..."

[models.fast]
provider = "openai-compat"
model     = "llama-3.3-70b-versatile"
base_url  = "https://api.groq.com/openai/v1"

[providers.mygateway]      # 방금 나온 서비스
kind        = "custom"     # JSONPath 매핑 방식
base_url    = "https://api.newsvc.ai/v2/chat"
auth_scheme = "bearer"
[mappings.newsvc]
text       = "result.choices[0].message.content"
tool_calls = "result.choices[0].message.tool_calls[*]"
usage_in   = "result.usage.prompt_tokens"
```

## 8. 확장 포인트 (기여 포인트)

| 기여 대상 | 파일 | 난이도 |
| --- | --- | --- |
| 새 LLM 서비스의 네이티브 어댑터 | `providers/<svc>.py` | 중 |
| 새 서비스 프리셋 등록 (설정만으로 충분) | `core/presets.py` | 하 |
| 새 내장 도구 | `tools/<tool>.py` | 하 |
| MCP 서버 연동 | 설정 `[[mcp_servers]]` | 없음 |

## 9. 테스트 전략

- **단위**: 라우터 모델 레퍼런스 파싱, 토큰 추정, 메시지 트리밍 경계 정합성,
  JSONPath 매핑, SSE 파서(개행 분할·중단·에러 이벤트).
- **계측**: `fake` 프로바이더로 전체 루프를 no-network로 검증
  (tool_call → 실행 → 재요청 → 종료 시나리오).
- **계약 테스트**: 어댑터 공통 인터페이스 준수 검증.

## 10. 로드맵

- [x] Phase 1 — 코어 타입/프로바이더 추상화, 라우터, 프리셋
- [x] Phase 2 — OpenAI 호환 어댑터 + SSE 스트리밍
- [x] Phase 3 — 도구 레지스트리 + 파일/셸/에이전트 도구
- [x] Phase 4 — 컨텍스트 관리(추정/압축/윈도)
- [x] Phase 5 — 에이전트 루프 + 권한
- [x] Phase 6 — 세션 영속화
- [x] Phase 7 — CLI/TUI
- [x] Phase 8 — MCP 클라이언트 (stdio/HTTP)
- [x] Phase 9 — Anthropic/Gemini/Azure/Ollama/커스텀 어댑터
- [ ] Phase 10 — 서브에이전트, 권한 고도화, 관측성

## Implementation status (2026-10-08)

Azure and Bedrock native transports remain planned. Their preset identifiers reject explicitly; they never fall through to an unrelated provider. Current native adapters are Anthropic, Gemini, and Ollama. See README.md and SECURITY.md for current permission, workspace, session and transport guarantees.

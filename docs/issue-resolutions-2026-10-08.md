# Issue resolution coverage (2026-10-08)

Regression tests were run before implementation and then rerun against the fixes. Tests use offline HTTP mocks; MCP lifecycle also has an actual local stdio JSON-RPC server integration test. Azure and Bedrock native adapters remain explicitly unsupported rather than silently routing to another provider.

| Issues | Change | Verification |
| --- | --- | --- |
| #9 | 각 스트리밍 도구 호출의 JSON 인자를 독립적으로 파싱하도록 수정했습니다. | `test_parallel_openai_arguments_are_independent` |
| #10 | 잘못된 도구 반환값을 도구 오류로 변환하여 턴 전체 실패를 방지했습니다. | `test_invalid_result_does_not_abort_registry` |
| #11 | 완료 이벤트의 사용량을 기준으로 토큰을 한 번만 집계합니다. | `test_complete_usage_is_authoritative_once` |
| #12 | MCP 이름에 서버 접두사를 기본 적용하고 중복 등록을 거부하며 제거 시 객체 소유권을 확인합니다. | `test_mcp_cannot_shadow_builtin_or_claim_readonly` |
| #13 | Anthropic thinking 서명, redacted 블록 및 원래 블록 순서를 보존해 재전송합니다. | `test_anthropic_signed_thinking_replayed_once; test_anthropic_interleaved_tools_preserve_block_order` |
| #14, #34 | OpenAI 루트에는 /v1을 보완하고 이미 버전이 지정된 제공자 경로는 유지합니다. 프리셋 이름과 별칭도 정상 조회합니다. | `test_api_root_normalization; test_bare_ollama_tag_and_preset_lookup` |
| #23 | 세션 ID와 최종 경로를 검증해 저장 디렉터리 밖의 파일에 접근하지 못하게 했습니다. | `tests/test_runner_session_regressions.py` |
| #24, #69 | 현재 사용자 요청과 이후 도구 호출 전체를 압축에서 보존하며, 입력 예산에 맞지 않으면 명시적 오류를 반환합니다. | `tests/test_runner_session_regressions.py` |
| #25 | 접두사 없는 Ollama 모델 태그를 공급자 접두사로 오인하지 않고 로컬 Ollama로 라우팅합니다. | `test_bare_ollama_tag_and_preset_lookup` |
| #26 | 사용자 정의 제공자의 분할 도구 인자를 누적 파싱하고 최종 파싱 실패를 명시적인 오류로 보고합니다. | `test_custom_fragmented_arguments; test_custom_malformed_final_arguments_are_terminal_error` |
| #27 | 사용자 정의 제공자 요청 이력에 assistant tool_calls와 결과의 tool_call_id를 보존합니다. | `test_custom_history_keeps_correlations` |
| #28 | 스트리밍 연결의 httpx 네트워크 오류를 ProviderError로 변환합니다. | `test_stream_transport_maps_network_failure` |
| #29 | 설정을 임시 파일에 원자적으로 저장하고 POSIX에서 소유자만 읽고 쓰도록 제한합니다. Windows ACL 상속과 환경 변수 사용을 SECURITY.md에 설명했습니다. | `test_config_failed_atomic_replace_preserves_original; test_config_is_owner_only` |
| #30 | 미구현 Azure/Bedrock 식별자가 다른 제공자로 조용히 전송되지 않도록 명시적으로 거부합니다. 네이티브 미지원 상태와 게이트웨이 대안을 문서화했습니다. | `test_unsupported_native_provider_rejected` |
| #31 | 원격 MCP의 readOnlyHint를 권한 근거로 신뢰하지 않고 execute 정책을 적용합니다. | `test_mcp_cannot_shadow_builtin_or_claim_readonly` |
| #32 | web_fetch의 기본 승인 정책을 ASK로 바꾸고 사설·로컬 목적지 및 리디렉션을 차단합니다. 검증한 공개 IP에 연결을 고정하고 원래 Host/TLS SNI를 유지합니다. | `test_web_fetch_blocks_private_and_credential_urls; test_web_redirect_private_target_rejected_before_second_request` |
| #33 | 셸 타임아웃을 유한한 양수와 최대 600초로 검증합니다. | `test_shell_rejects_unbounded_timeout` |
| #35 | Retry-After의 초 단위 값과 HTTP 날짜를 파싱해 속도 제한 재시도에 반영합니다. | `test_retry_after_populated` |
| #36 | 도구명·인자·출력·오류·세션 제목 등 외부 문자열을 Rich 마크업으로 해석하지 않고 그대로 표시합니다. | `test_tui_renders_untrusted_markup_literally; test_sessions_display_untrusted_titles_literally` |
| #37 | run_turn_to_completion에서 ErrorEvent를 예외로 전달해 빈/부분 응답을 성공으로 반환하지 않습니다. | `tests/test_runner_session_regressions.py` |
| #38 | 프리셋의 본문 기본값, 프로토콜 및 출력 한도를 실제 요청에 반영하고 runner도 기본 출력량을 컨텍스트 예산에서 예약합니다. | `test_preset_defaults_and_protocol_apply; test_heuristic_provider_uses_preset_output_defaults; test_runner_reserves_provider_default_output_budget` |
| #39 | 세션 목록에 전체 ID를 표시하고 재개 시 정확한 ID 또는 유일한 접두사를 안전하게 해석합니다. | `test_sessions_display_untrusted_titles_literally; tests/test_runner_session_regressions.py` |
| #40 | 기존 형식의 세션도 재개할 때 현재 작업공간과 사용자 지침으로 생성된 시스템 프롬프트를 갱신합니다. 명시적으로 전달한 사용자 시스템 메시지는 보존합니다. | `tests/test_runner_session_regressions.py` |
| #41 | MCP의 전체 중첩 input_schema를 보존하고 각 제공자 요청 스키마에 그대로 전달합니다. | `test_mcp_schema_is_preserved; test_full_nested_tool_schema_survives` |
| #42, #44 | 터미널 확인 질문과 상태를 변경하는 도구를 각각 직렬화해 중복 프롬프트와 쓰기 순서 경쟁을 방지합니다. | `test_mutations_and_confirmations_keep_order` |
| #43, #73 | SIGINT로 현재 REPL 턴만 취소하고, 취소·GeneratorExit·도구 예외 시 대화 이력을 이전 정상 상태로 복구합니다. | `test_repl_sigint_cancels_only_active_turn; tests/test_runner_session_regressions.py` |
| #45, #75 | Windows 출력은 UTF-8을 우선 확인하고 OEM 코드 페이지로 폴백합니다. PowerShell UTF-8 출력과 닫힌 stdin도 적용했습니다. | `test_windows_oem_output_fallback; test_shell_strips_secrets_and_closes_stdin` |
| #46 | 파일을 원자적으로 쓰며 LF/CRLF와 BOM을 보존합니다. LF로 작성한 여러 줄 검색도 CRLF 파일에서 편집할 수 있습니다. | `test_fs_preserves_newlines_and_deletes_moves; test_multiline_lf_edit_preserves_crlf_and_bom; test_atomic_edit_failure_preserves_original_and_cleans_temp` |
| #47, #76 | 파일 읽기·검색·목록의 입력/출력/탐색 상한을 적용하고 .git·.venv·의존성 디렉터리와 바이너리를 제외합니다. | `test_scans_exclude_dependencies_binaries_and_support_single_file` |
| #48 | 응답을 스트리밍으로 읽으며 2MB 상한에서 중단하고 압축 응답으로 상한을 우회하지 못하게 했습니다. | `test_web_stream_download_limit_stops_reading` |
| #49 | 도구 실행 전에 전체 JSON Schema로 인자 타입·필수값·열거값·중첩 제약을 검증합니다. | `test_arguments_validated_before_execution` |
| #50 | Linux/Windows 및 Python 3.11–3.13 CI를 추가하고 패키지 링크/불필요 의존성을 정리했습니다. CHANGELOG·CONTRIBUTING·SECURITY 문서를 추가했습니다. | `.github/workflows/ci.yml; wheel/sdist build` |
| #52 | 셸 자식 환경을 필수 OS 변수 허용 목록으로 제한해 제공자 인증 변수의 상속을 차단합니다. | `test_shell_strips_secrets_and_closes_stdin` |
| #53 | 셸 편의 함수의 workspace_root를 요청 cwd와 독립적으로 지정·검증합니다. | `test_shell_helper_root_independent_of_cwd` |
| #54 | 음수 max_items를 거부하고 목록 출력 개수를 상한으로 제한합니다. | `test_scans_exclude_dependencies_binaries_and_support_single_file` |
| #55 | 손상된 JSONL 행은 위치를 기록하며 건너뛰고 정상 메시지를 복원합니다. 유효하지 않은 도구 대응 그룹은 제거합니다. | `tests/test_runner_session_regressions.py` |
| #56 | 설정의 상대 workspace 및 MCP cwd를 해당 TOML 파일 디렉터리를 기준으로 해석합니다. | `test_config_relative_paths_use_config_directory` |
| #57 | 정상 종료 및 반복 한도 종료에서 세션을 저장한 후 DoneEvent를 전달합니다. | `tests/test_runner_session_regressions.py` |
| #58, #59 | TOML 호환 Unicode 직렬화로 이모지를 보존하며 MCP headers·timeout·read_timeout 등 모든 설정을 저장/재로드합니다. | `test_config_round_trip_unicode_and_mcp_options; test_config_mcp_options_round_trip_without_unicode` |
| #61 | 0 이하 chunk_size를 즉시 거부하여 무한 반복을 방지합니다. | `test_invalid_text_chunk_size_rejected` |
| #65 | 도구 호출 ID와 결과 ID의 완전한 일대일 대응을 검증하며 중복·누락·잘못된 결과를 요청에 전달하지 않습니다. | `tests/test_runner_session_regressions.py` |
| #67 | 인증된 제공자 요청은 리디렉션을 따라가지 않으므로 다른 출처에 사용자 인증 헤더가 전달되지 않습니다. | `test_redirect_never_forwards_credentials` |
| #68 | 빈 OPENAGENT_SESSION_DIR은 현재 디렉터리가 아니라 기본 세션 저장 위치를 사용합니다. | `tests/test_runner_session_regressions.py` |
| #70 | 연결 실패 후 다음 작업에서 재연결·도구 재탐색·레지스트리 갱신과 백오프를 수행합니다. 실패한 작업을 자동 재실행하지 않으며, AnyIO 연결 수명은 전용 소유 태스크에서 관리합니다. | `test_client_next_call_reconnects_and_relists; test_manager_refreshes_tools_after_disconnect_on_next_operation; test_real_stdio_mcp_reconnect_across_tasks` |
| #71 | 도구 설명·결과 및 압축 이력에 명시적인 비신뢰 데이터 경계와 지침 우선순위를 적용합니다. | `tests/test_runner_session_regressions.py; tests/test_tool_safety_regressions.py` |
| #72 | Gemini의 연속 사용자/도구 결과를 하나의 contents 항목으로 합쳐 도구 병렬 호출 뒤 역할 순서를 유지합니다. | `test_gemini_groups_parallel_results` |
| #74 | Ollama thinking 필드와 분할된 inline <think> 내용을 ThinkingDelta 및 저장 메시지의 reasoning으로 보존합니다. | `test_ollama_thinking_preserved; test_ollama_inline_thinking_split_across_chunks` |
| #77 | sessions delete/rm, prune --older-than, --clear를 추가하고 정확/유일 접두사, 확인 질문, 잘못된 기간 입력 거부를 지원합니다. | `test_sessions_delete_by_prefix; test_sessions_prune; tests/test_runner_session_regressions.py` |
| #78 | 확인 대상 write 권한의 delete_file/move_file을 등록했습니다. 작업공간 경계를 검증하고 디렉터리 삭제와 기존 파일 덮어쓰기를 거부합니다. | `test_fs_preserves_newlines_and_deletes_moves` |
| #79 | AGENTS.md·CLAUDE.md·.openagent/rules.md 등의 지침을 크기와 심볼릭 링크 경계를 검사해 자동 로드합니다. | `tests/test_runner_session_regressions.py` |
| #80 | REPL에 모델 전환, 누적 사용량/컨텍스트 확인, 도구 표와 triple-quote 여러 줄 입력을 추가했습니다. | `tests/test_repl_commands.py` |

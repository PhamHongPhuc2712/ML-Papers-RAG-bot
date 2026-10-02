# Recorded chat-completion responses

The chat-client tests (`tests/unit/test_llm_client.py`) parse these bodies through
`httpx.MockTransport`; no test touches the network.

| File | Source | Edited by hand |
|---|---|---|
| `openai_chat.json` | A real `gpt-6-luna` response, captured 2026-10-02 (L2 step 3): message "Reply with the word ok.", `reasoning_effort: none`, `max_completion_tokens: 1024` | `id` → `"fixture"`; `usage.prompt_tokens` 12 → 2048, `usage.prompt_tokens_details.cached_tokens` 0 → 1024 and `usage.total_tokens` 16 → 2052, so the cached-input parsing is pinned |
| `deepseek_chat.json` | **Not captured**: no DeepSeek key was configured. Written from the non-streaming response example in DeepSeek's API reference (api-docs.deepseek.com/api/create-chat-completion, read 2026-10-02) | `id` → `"fixture"`; `content` → `"ok"`; usage set to 1,536 prompt tokens with `prompt_cache_hit_tokens` 512 and `prompt_cache_miss_tokens` 1,024 |

Replace `deepseek_chat.json` with a captured body before `deepseek-flash` is used, since it
is the pilot's fallback only.

What the capture established (configs/llm.yaml records it):
- `gpt-6-luna` serves under its alias, `"model": "gpt-6-luna"`, with no dated snapshot.
- At effort `none` it accepts `temperature: 0` and `seed: 42`, alone and together.
- At effort `low` it refuses `temperature: 0` (HTTP 400) and accepts `seed: 42`.
- `completion_tokens` includes `completion_tokens_details.reasoning_tokens` (0 at `none`).

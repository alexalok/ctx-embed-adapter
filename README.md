# ctx OpenRouter embedding adapter

Local Node.js 26 service implementing the ctx V2 contract. When selected as a ctx executor, it sends document chunks and search queries to OpenRouter for embedding. It does not persist text or vectors on disk. Loopback access is unauthenticated; other local processes can use the service and consume API credit.

Run from this directory:

```sh
/opt/homebrew/bin/node server.mjs
```

The service reads `OPENROUTER_API_KEY` from its environment, or from the env file named by `OPENROUTER_ENV_FILE` (no default). Only the key assignment is parsed; the file is never executed. `GET /v2/contract` reports the space identity. `GET /healthz` reports process status and cumulative request, upstream call, cache hit and reported token counts since startup; it does not probe OpenRouter. Request logs include metadata only. SIGTERM drains active requests.

| Environment variable | Default |
| --- | --- |
| CTX_EMBED_HOST | 127.0.0.1 (literal loopback only; IPv6 ::1 also supported) |
| CTX_EMBED_PORT | 8791 |
| CTX_EMBED_MODEL | mongodb/voyage-4 |
| CTX_EMBED_DIMENSIONS | 1024 |
| CTX_EMBED_SPACE_ID | ctx-embed-adapter/openrouter/<model>/<dimensions>/v1 |
| CTX_EMBED_BATCH_SIZE | 64 |
| CTX_EMBED_CONCURRENCY | 4 (process-wide upstream calls) |
| CTX_EMBED_TIMEOUT_MS | 20000 (per upstream attempt; 22 s aggregate budget) |
| CTX_EMBED_MAX_CHARS | 100000 (UTF-16 code units before byte cap) |

[OpenRouter embeddings](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings) accepts `dimensions`, `encoding_format: "float"` and `input_type`. For [Voyage 4](https://openrouter.ai/mongodb/voyage-4), the adapter sends `query` or `document`, with no prefix. [Voyage limits](https://docs.voyageai.com/docs/embeddings) include 1,000 inputs, 32K tokens per input and 320K tokens per request for this model. The adapter uses conservative additional limits of 30,000 UTF-8 bytes per input and 250,000 bytes per upstream batch; this bounds even token-dense inputs without a tokenizer. Blank text maps to `[empty]`. Change space ID before accepting incompatible changes in model, preprocessing or vector semantics.

Successful responses are reused within a process-local LRU (128 entries, 32 MiB). Concurrent duplicates coalesce; changed bytes with a retained ID give 409. Failed requests retain their body hash and allow retry. Eviction or restart forgets prior IDs. Multiple batches can finish before a sibling fails, so retries of failed requests can incur usage for those batches again. Totals count usage returned by OpenRouter, not invoice data.

```sh
/opt/homebrew/bin/node --test test.mjs
CTX_EMBED_SMOKE=1 /opt/homebrew/bin/node --test test.mjs
```

The second command also sends the same three short synthetic strings in both modes, checks dimensions and norms, and prints query/document cosine similarities and reported token use. Neither command reads ctx history or config.

## Install as a launchd agent (macOS)

Copy `ctx-embed-adapter.plist.example`, replace `YOUR_USER` in its absolute paths, and point `OPENROUTER_ENV_FILE` at a file containing `OPENROUTER_API_KEY=...`. It expects `server.mjs` in `~/.local/share/ctx-embed-adapter` and logs to `~/Library/Logs/ctx-embed-adapter.log`.

```sh
mkdir -p ~/.local/share/ctx-embed-adapter
cp server.mjs ~/.local/share/ctx-embed-adapter/
cp ctx-embed-adapter.plist.example ~/Library/LaunchAgents/local.ctx-embed-adapter.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.ctx-embed-adapter.plist
ctx semantic enable --executor http://127.0.0.1:8791
```

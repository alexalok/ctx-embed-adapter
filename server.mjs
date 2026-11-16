import http from 'node:http';
import { isIP } from 'node:net';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';

const MIB = 1024 * 1024;
const MODEL = 'mongodb/voyage-4';
const fail = (status, message) => Object.assign(new Error(message), { status });
export function config(env = process.env) {
  function num(name, fallback, max) {
    const n = Number(env['CTX_EMBED_' + name] ?? fallback);
    if (!Number.isSafeInteger(n) || n < 1 || n > max) throw Error('Invalid CTX_EMBED_' + name);
    return n;
  }
  const host = env.CTX_EMBED_HOST ?? '127.0.0.1';
  if (!(host === '::1' || (isIP(host) === 4 && host.startsWith('127.')))) throw Error('Literal loopback address required');
  const model = env.CTX_EMBED_MODEL ?? MODEL;
  const dimensions = num('DIMENSIONS', 1024, 4096);
  const space_id = env.CTX_EMBED_SPACE_ID ?? ('ctx-embed-adapter/openrouter/' + model + '/' + dimensions + '/v1');
  if (!space_id || space_id.length > 256 || /[^A-Za-z0-9._:/@+=-]/.test(space_id)) throw Error('Invalid space_id');
  return { host, port: num('PORT', 8791, 65535), model, space_id, dimensions,
    batch: num('BATCH_SIZE', 64, 512), concurrency: num('CONCURRENCY', 4, 64),
    timeout: num('TIMEOUT_MS', 20000, 22000), chars: num('MAX_CHARS', 100000, 8 * MIB) };
}
export function apiKey(env = process.env) {
  if (env.OPENROUTER_API_KEY?.trim()) return env.OPENROUTER_API_KEY.trim();
  if (env.OPENROUTER_ENV_FILE) try {
    const file = readFileSync(env.OPENROUTER_ENV_FILE, 'utf8');
    const m = file.match(/^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(?:"([^"\r\n]+)"|'([^'\r\n]+)'|([^\s#'"]+))[ \t]*(?:#.*)?\r?$/m);
    if (m) return m[1] ?? m[2] ?? m[3];
  } catch {}
  throw Error('Missing OPENROUTER_API_KEY; check env or OPENROUTER_ENV_FILE');
}
function prepare(text, chars) {
  // Bound UTF-8 bytes as well as characters; characters alone cannot bound tokens.
  const buf = Buffer.from(text.slice(0, chars).toWellFormed());
  let end = Math.min(30000, buf.length);
  while (end < buf.length && (buf[end] & 0xc0) === 0x80) end--;
  const prefix = buf.toString('utf8', 0, end);
  return prefix.trim() ? prefix : '[empty]';
}
async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req.iterator({ destroyOnReturn: false })) {
    size += chunk.length;
    if (size > 8 * MIB) { req.resume(); throw fail(413, 'Body too large'); }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}
function normalize(v, dim) {
  if (!Array.isArray(v) || v.length !== dim || !v.every(Number.isFinite)) throw fail(502, 'Invalid upstream vector');
  const norm = Math.hypot(...v);
  if (!norm || !Number.isFinite(norm)) throw fail(502, 'Invalid upstream norm');
  return v.map(x => x / norm);
}

export function createAdapter({ cfg = config(), key = apiKey(), fetchFn = fetch, log = console.log } = {}) {
  const contract = { schema_version: 2, space_id: cfg.space_id, dimensions: cfg.dimensions };
  const totals = { requests: 0, upstream_calls: 0, tokens: 0, cache_hits: 0 };
  const cache = new Map(), pending = new Map();
  let bytes = 0, slots = 0;
  function remember(id, item) {
    if (cache.has(id)) { bytes -= cache.get(id).size; cache.delete(id); }
    item.size = Buffer.byteLength(item.result ?? '') + Buffer.byteLength(id) + 64;
    bytes += item.size;
    cache.set(id, item);
    while (cache.size > 128 || bytes > 32 * MIB) {
      const oldest = cache.keys().next().value;
      bytes -= cache.get(oldest).size;
      cache.delete(oldest);
    }
  }
  async function embed(req, stats, deadline) {
    const stop = new AbortController();
    const signal = AbortSignal.any([stop.signal, AbortSignal.timeout(Math.max(1, deadline - Date.now()))]);
    const groups = [];
    let group = [], length = 0;
    for (const input of req.inputs) {
      const value = prepare(input.text, cfg.chars), n = Buffer.byteLength(value);
      if (group.length >= cfg.batch || length + n > 250000) { groups.push(group); group = []; length = 0; }
      group.push({ id: input.id, value }); length += n;
    }
    groups.push(group);
    const embeddings = [];
    async function run() {
      while (groups.length) {
        const batch = groups.shift();
        while (slots >= cfg.concurrency) await sleep(10, null, { signal });
        signal.throwIfAborted(); slots++;
        try {
          const body = JSON.stringify({ model: cfg.model, input: batch.map(i => i.value), encoding_format: 'float',
            dimensions: cfg.dimensions, input_type: req.input_kind === 'query' ? 'query' : 'document' });
          for (let attempt = 0;; attempt++) {
            totals.upstream_calls++;
            const res = await fetchFn('https://openrouter.ai/api/v1/embeddings', { method: 'POST', redirect: 'error',
              headers: { authorization: 'Bearer ' + key, 'content-type': 'application/json' }, body,
              signal: AbortSignal.any([signal, AbortSignal.timeout(cfg.timeout)]) });
            if (!res.ok) {
              await res.body?.cancel();
              const retryable = res.status === 429 || res.status >= 500;
              const wait = Math.max(100, Number(res.headers.get('retry-after') ?? 0) * 1000);
              if (!attempt && retryable && wait <= 500 && Date.now() + wait + 1000 < deadline) {
                await sleep(wait, null, { signal }); continue;
              }
              throw fail(res.status === 429 ? 429 : retryable ? 503 : 500, 'OpenRouter HTTP ' + res.status);
            }
            const json = await res.json();
            const tokens = json?.usage?.total_tokens ?? json?.usage?.prompt_tokens;
            if (Number.isSafeInteger(tokens) && tokens >= 0) { stats.tokens += tokens; totals.tokens += tokens; }
            if (!Array.isArray(json?.data) || json.data.length !== batch.length) throw fail(502, 'Invalid upstream count');
            const seen = new Set();
            for (const row of json.data) {
              const i = row?.index;
              if (!Number.isInteger(i) || i < 0 || i >= batch.length || seen.has(i)) throw fail(502, 'Invalid upstream index');
              seen.add(i); embeddings.push({ id: batch[i].id, embedding: normalize(row.embedding, cfg.dimensions) });
            }
            break;
          }
        } finally { slots--; }
      }
    }
    const started = Date.now();
    try {
      await Promise.all(Array.from({ length: Math.min(groups.length, cfg.concurrency) }, run));
      const result = JSON.stringify({ ...contract, request_id: req.request_id, embeddings });
      if (Buffer.byteLength(result) > 8 * MIB) throw fail(413, 'Response too large');
      return result;
    } catch (e) { throw e.status ? e : fail(503, 'OpenRouter transport or timeout'); }
    finally { stop.abort(); stats.upstream_ms = Date.now() - started; }
  }
  return http.createServer({ requestTimeout: 23000, headersTimeout: 5000 }, async (req, res) => {
    const started = Date.now();
    let result, status = 200;
    const stats = { request_id: null, kind: null, count: 0, chars: 0, upstream_ms: 0, tokens: 0 };
    totals.requests++;
    try {
      const local = req.socket.localAddress;
      const address = isIP(local) === 6 ? '[' + local + ']' : local;
      if (req.headers.origin || ![address, address + ':' + req.socket.localPort].includes(req.headers.host))
        throw fail(403, 'Literal loopback Host and non-browser request required');
      if (req.url === '/healthz' && req.method === 'GET') result = JSON.stringify({status: 'ok', model: cfg.model, ...contract, totals});
      else if (req.url === '/v2/contract' || req.url === '/v2/embeddings') {
        const version = req.headers['x-ctx-semantic-schema-version'];
        if (version !== undefined && version !== '2') throw fail(400, 'Schema mismatch');
        if (req.method === 'GET' && req.url === '/v2/contract') result = JSON.stringify(contract);
        else if (req.method === 'POST' && req.url === '/v2/embeddings') {
          if (req.headers['content-type']?.split(';')[0] !== 'application/json') throw fail(415, 'Expected application/json');
          const raw = await readBody(req);
          let r;
          try { r = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(raw)); } catch { throw fail(400, 'Invalid JSON'); }
          if (!r || !Object.entries(contract).every(([k,v]) => r[k] === v)) throw fail(400, 'Contract mismatch');
          if (typeof r.request_id !== 'string' || !r.request_id || !['query','documents'].includes(r.input_kind)
            || !Array.isArray(r.inputs) || !r.inputs.length || r.inputs.length > Math.min(512, Math.floor(262144/cfg.dimensions)))
            throw fail(400, 'Invalid embedding request');
          const ids = new Set();
          for (const input of r.inputs) {
            if (typeof input?.id !== 'string' || typeof input.text !== 'string' || ids.has(input.id)) throw fail(400, 'Invalid input');
            ids.add(input.id); stats.chars += input.text.length;
          }
          const id = r.request_id;
          Object.assign(stats, { request_id: id.slice(0, 128), kind: r.input_kind, count: ids.size });
          const hash = createHash('sha256').update(raw).digest('hex');
          const item = pending.get(id) ?? cache.get(id);
          if (item && item.hash !== hash) throw fail(409, 'Request ID reused with different body');
          if (item?.result || item?.promise) {
            totals.cache_hits++;
            if (item.result) remember(id, item);
            result = item.result ?? await item.promise;
          } else {
            if (pending.size >= 8) throw fail(503, 'Adapter busy');
            const promise = embed(r, stats, started + 22000);
            pending.set(id, { hash, promise });
            try { result = await promise; }
            finally { remember(id, {hash, result}); pending.delete(id); }
          }
        } else throw fail(405, 'Method not allowed');
      } else throw fail(404, 'Not found');
    } catch (e) {
      status = e.status ?? 500;
      stats.error = e.status ? e.message : 'Internal error';
      result = JSON.stringify({ error: stats.error });
    }
    res.writeHead(status, { 'content-type': 'application/json', 'cache-control': 'no-store' }).end(result);
    log(JSON.stringify({ ...stats, status }));
  });
}
if (import.meta.main) {
  const cfg = config(), server = createAdapter({ cfg });
  server.listen(cfg.port, cfg.host);
  process.once('SIGTERM', () => {
    server.close();
    setTimeout(() => server.closeAllConnections(), 23000).unref();
  });
}

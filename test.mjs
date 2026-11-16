import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { config, apiKey, createAdapter } from './server.mjs';

async function fixture(t, opts) {
  const server = createAdapter(opts);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = 'http://127.0.0.1:' + server.address().port;
  const get = path => fetch(base + path).then(r => r.json());
  const contract = await get('/v2/contract');
  const payload = { ...contract, request_id: 'abc', input_kind: 'documents', inputs: [
    {id:'first', text:'solid'}, {id:'second', text:'  '}] };
  const post = (r, suffix = '') => fetch(base + '/v2/embeddings', {method:'POST', headers:{'content-type':'application/json',
    'x-ctx-semantic-schema-version':'2'}, body:JSON.stringify(r) + suffix});
  return {contract,payload,post,get};
}
const mock = fetchFn => ({cfg:config({CTX_EMBED_DIMENSIONS:'2'}), key:'test', fetchFn, log(){}});
test('contract, IDs, unit norm, blank text, concurrent duplicates and replay', async t => {
  let calls = 0;
  const {contract,payload,post,get} = await fixture(t, mock(async (_url, init) => {
    calls++;
    const body = JSON.parse(init.body);
    assert.deepEqual(body.input,['solid','[empty]']);
    assert.equal(body.input_type,'document');
    assert.equal(body.dimensions,2);
    await new Promise(r => setTimeout(r,30));
    return Response.json({usage:{total_tokens:9},data:[{index:1,embedding:[0,5]},{index:0,embedding:[3,4]}]});
  }));
  assert.deepEqual(contract,{ schema_version:2, space_id:'ctx-embed-adapter/openrouter/mongodb/voyage-4/2/v1', dimensions:2 });
  const [a,b] = await Promise.all([post(payload),post(payload)]);
  assert.equal(a.status,200);
  const raw = await a.text();
  assert.equal(await b.text(),raw);
  assert.equal(await (await post(payload)).text(),raw);
  assert.deepEqual(Object.fromEntries(JSON.parse(raw).embeddings.map(i=>[i.id,i.embedding])), {first:[0.6,0.8],second:[0,1]});
  assert.equal((await post(payload,' ')).status,409);
  assert.equal(calls,1);
  assert.equal((await get('/healthz')).totals.tokens,9);
});
test('mismatched identity, duplicate IDs, input count bounds rejected before upstream', async t => {
  let calls=0;
  const {post,payload} = await fixture(t,mock(()=>calls++));
  for (const change of [{space_id:'other'},{dimensions:3},{schema_version:1},{inputs: Array(513).fill(payload.inputs[0])},
    {inputs:Array(2).fill(payload.inputs[0])}]) assert.equal((await post({...payload,...change})).status,400);
  assert.equal(calls,0);
});
test('429 retries once, preserves body hash for next ctx retry', async t => {
  let calls=0;
  const {post,payload} = await fixture(t,mock(async()=>{calls++;return new Response(null,{status:429});}));
  assert.equal((await post(payload)).status,429);
  assert.equal(calls,2);
  assert.equal((await post(payload,' ')).status,409);
  assert.equal((await post(payload)).status,429);
  assert.equal(calls,4);
});
test('zero vector, missing dimensions, duplicate index rejected', async t => {
  let data;
  const {post,payload} = await fixture(t,mock(()=>Response.json({data})));
  for (const [i,v] of [[1,[0,0]],[1,[1]],[0,[1,1]]]) {
    data = [{index:0,embedding:[3,4]},{index:i,embedding:v}];
    assert.equal((await post(payload)).status,502);
  }
});
test('remote host rejected',()=>assert.throws(()=>config({CTX_EMBED_HOST:'0.0.0.0'}),/loopback/));
test('live OpenRouter, three synthetic inputs in both modes', {skip:process.env.CTX_EMBED_SMOKE !== '1'}, async t=>{
  const {post,payload,get} = await fixture(t,{cfg:config({}),key:apiKey()});
  const inputs=['A kettle boils water.','Blueberry muffins.','An oak tree.'].map((text,i)=>({id:String(i),text}));
  let previous;
  for(const input_kind of ['query','documents']) {
    const res=await post({...payload,request_id:'smoke-'+input_kind,input_kind,inputs});
    const r=await res.json();
    assert.equal(res.status,200,JSON.stringify(r.error));
    const rows=r.embeddings.sort((a,b)=>Number(a.id)-Number(b.id));
    assert.deepEqual(rows.map(i=>i.id),['0','1','2']);
    const norms=rows.map(i=>i.embedding.reduce((s,x)=>s+x*x,0));
    assert.ok(rows.every(i=>i.embedding.length===1024) && norms.every(n=>Math.abs(n-1)<0.001));
    t.diagnostic(JSON.stringify({input_kind,dimensions:1024,squared_norms:norms}));
    if(previous) t.diagnostic('same-text cosine (query vs document): '+rows.map((r,i)=>r.embedding.reduce((s,x,j)=>s+x*previous[i].embedding[j],0)).join(', '));
    previous=rows;
  }
  t.diagnostic('reported tokens: '+(await get('/healthz')).totals.tokens);
});

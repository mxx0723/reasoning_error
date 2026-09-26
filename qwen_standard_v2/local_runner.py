"""Local open weights: V4 Flash via offline vLLM; R1 Distill/Qwen via Transformers."""
import argparse
import json
import os
from pathlib import Path
import signal
from pipeline import Pipeline,VERSION,digest
from hf_backend import HFBackend,CachedModel,load_sources,checkpoint_identity


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--profile',choices=['deepseek-v4','deepseek-r1','qwen'],default='deepseek-v4')
    p.add_argument('--encoder-path',type=Path,help='Official V4 encoding_dsv4.py in the checkpoint encoding directory')
    p.add_argument('--thinking-mode',choices=['chat','thinking'],default='chat')
    p.add_argument('--max-model-len',type=int,default=32768)
    p.add_argument('--engine-args',type=Path,help='JSON object of offline vLLM LLM engine parameters')
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ids');g=p.add_mutually_exclusive_group();g.add_argument('--limit',type=int,default=1);g.add_argument('--all',action='store_true')
    p.add_argument('--max-memory',default='{}',help='JSON, e.g. {"0":"42GiB","1":"42GiB"}')
    p.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    p.add_argument('--temperature',type=float);p.add_argument('--top-p',type=float)
    p.add_argument('--max-new-tokens',type=int,default=8192);p.add_argument('--max-input-tokens',type=int,default=16384)
    p.add_argument('--seed',type=int,default=20260926);p.add_argument('--max-calls-per-source',type=int,default=400)
    p.add_argument('--history',choices=['final','full'],default='final')
    p.add_argument('--allow-cpu-offload',action='store_true');p.add_argument('--trust-remote-code',action='store_true')
    p.add_argument('--reviews',type=Path);p.add_argument('--seed-plans',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    temperature=a.temperature if a.temperature is not None else {'deepseek-v4':1.,'deepseek-r1':.6,'qwen':.7}[a.profile]
    top_p=a.top_p if a.top_p is not None else (.95 if a.profile=='deepseek-r1' else 1.)
    if temperature<=0 or not 0<top_p<=1 or min(a.limit,a.max_new_tokens,a.max_input_tokens,a.max_calls_per_source)<=0:
        p.error('Invalid sampling parameters or limits')
    memory=json.loads(a.max_memory)
    if not isinstance(memory,dict):p.error('max-memory must be a JSON object')
    rows=load_sources(a.data)
    if a.ids:
        ids=set(a.ids.split(','));rows=[s for s in rows if s['id'] in ids]
        if {s['id'] for s in rows}!=ids:p.error('Requested IDs missing from data')
    if not a.all:rows=rows[:a.limit]
    if not rows:p.error('No selected sources')
    reviews=json.loads(a.reviews.read_text(encoding='utf-8')) if a.reviews else {}
    plans=json.loads(a.seed_plans.read_text(encoding='utf-8')) if a.seed_plans else {}
    config=dict(model_path=str(a.model_path.resolve()),profile=a.profile,dtype=a.dtype,max_memory=memory,
                temperature=temperature,top_p=top_p,max_new_tokens=a.max_new_tokens,max_input_tokens=a.max_input_tokens,
                seed=a.seed,history=a.history,allow_cpu_offload=a.allow_cpu_offload,trust_remote_code=a.trust_remote_code)
    if a.profile=='deepseek-v4':
        if memory or a.allow_cpu_offload or a.dtype!='bfloat16':
            p.error('V4 uses engine-args for memory/dtype/offload, not Transformers flags')
        encoder=(a.encoder_path or a.model_path/'encoding'/'encoding_dsv4.py').resolve()
        if not encoder.is_file():p.error('V4 requires the official local encoding_dsv4.py')
        engine_args=json.loads(a.engine_args.read_text(encoding='utf-8')) if a.engine_args else {}
        if not isinstance(engine_args,dict):p.error('engine-args must be a JSON object')
        if a.max_model_len<a.max_input_tokens+a.max_new_tokens:p.error('max-model-len smaller than input plus output budgets')
        config.update(encoder_path=str(encoder),encoder_sha256=digest(encoder.read_text(encoding='utf-8')),
                      engine_args=engine_args,thinking_mode=a.thinking_mode,max_model_len=a.max_model_len)
        config.pop('dtype');config.pop('max_memory');config.pop('allow_cpu_offload')
    identity=checkpoint_identity(a.model_path)
    root=Path(__file__).parent
    codefiles=['pipeline.py','hf_backend.py','local_runner.py']+(['v4_backend.py'] if a.profile=='deepseek-v4' else [])
    manifest=dict(protocol=VERSION,backend='local_vllm_v4' if a.profile=='deepseek-v4' else 'local_transformers',config=config,checkpoint=identity,
                  source_ids=[s['id'] for s in rows],sources_sha256=digest(rows),seed_plans_sha256=digest(plans),
                  code_sha256={f:digest((root/f).read_text(encoding='utf-8')) for f in codefiles})
    if a.dry_run:print(json.dumps(manifest,ensure_ascii=True,indent=2));return
    a.output.mkdir(parents=True,exist_ok=True);lock=a.output/'active.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    def stop(signum,frame):raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM,stop)
    try:
        mf=a.output/'manifest.json'
        if mf.exists() and json.loads(mf.read_text(encoding='utf-8'))!=manifest:
            raise ValueError('Configuration, checkpoint, code or sources changed; use a new output directory')
        save(mf,manifest)
        if a.profile=='deepseek-v4':
            from v4_backend import V4Backend
            backend=V4Backend(config)
        else:backend=HFBackend(config)
        save(a.output/'runtime.json',backend.runtime)
        results=[]
        for source in rows:
            folder=a.output/source['id'];folder.mkdir(exist_ok=True)
            client=CachedModel(backend,folder,a.max_calls_per_source)
            engine=Pipeline(client.ask,client.raw,lambda name,value:save(folder/name,value),reviews,plans)
            try:
                engine.run_source(source);results.append(engine.summary())
                save(folder/'status.json',dict(status='completed',calls=len(client.cache),review_sha256=digest(reviews)))
            except BaseException as exc:
                engine.checkpoint();engine.summary()
                save(folder/'status.json',dict(status='failed_or_incomplete',error=str(exc)));raise
        save(a.output/'summaries.json',results)
        print('Completed',len(rows),'sources. Clinical review still required.')
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()

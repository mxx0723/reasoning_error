"""GPU entry point; one source per array task. Does not submit Slurm jobs."""
import hashlib
import json
import os
import signal
from pathlib import Path
import time
import traceback
from pipeline import Pipeline, VERSION, digest


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    source = json.loads((root/'sources.json').read_text(encoding='utf-8'))[index]
    output = root/'results'/source['id']; output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        path = output/name; tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'); tmp.replace(path)
    # Prevent two retries writing the same source concurrently. Remove only our lock.
    lock = output/'active.lock'
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps(dict(pid=os.getpid(), job=os.getenv('SLURM_JOB_ID'))).encode()); os.close(fd)
    engine = None
    try:
        def stop(signum, frame):
            raise KeyboardInterrupt('Scheduler termination signal')
        signal.signal(signal.SIGTERM, stop)
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        assert torch.cuda.device_count()==2, 'Two GPUs required by this deployment'
        torch.set_num_threads(4)
        tok = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(config['model_path'], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='auto',
            max_memory={0:'42GiB',1:'42GiB'}).eval()
        assert all(str(v) not in ('cpu','disk') for v in model.hf_device_map.values())
        save('runtime.json', dict(protocol=VERSION, source_sha256=digest(source), config=config,
             script_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['runner.py','pipeline.py']},
             job=os.getenv('SLURM_JOB_ID'), node=os.getenv('SLURMD_NODENAME'),
             gpus=[torch.cuda.get_device_name(i) for i in range(2)], started=time.time()))
        cache = {}; calls = output/'calls.jsonl'
        if calls.exists():
            for line in calls.read_text(encoding='utf-8').splitlines():
                row=json.loads(line); cache[row['key']]=row
        def ask(key, messages, limit=350, sample=True):
            spec=dict(messages=messages, limit=limit, sample=sample, protocol=VERSION,
                      model=config['model_path'], temperature=config['temperature'], seed=config['seed'],
                      top_p=1.0, top_k=0)
            signature=digest(spec)
            if key in cache:
                if cache[key]['signature']!=signature: raise ValueError('Cache mismatch: '+key)
                return cache[key]['parsed']
            if len(cache)>=config['max_calls_per_source']: raise RuntimeError('Call budget exhausted; incomplete')
            text=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs=tok(text, add_special_tokens=False, return_tensors='pt').to(model.get_input_embeddings().weight.device)
            if inputs.input_ids.shape[1]>config['max_input_tokens']: raise ValueError('Input too long; no truncation')
            seed=(config['seed']+int(hashlib.sha256(key.encode()).hexdigest()[:8],16))%(2**31)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            kwargs=dict(max_new_tokens=limit, do_sample=sample, use_cache=True, pad_token_id=tok.eos_token_id,
                        temperature=config['temperature'] if sample else 1., top_p=1., top_k=0)
            with torch.inference_mode(): tokens=model.generate(**inputs, **kwargs)[0,inputs.input_ids.shape[1]:]
            raw=tok.decode(tokens, skip_special_tokens=True); parsed=None
            if len(tokens)<limit:
                try: parsed=json.loads(raw[raw.index('{'):raw.rindex('}')+1])
                except (ValueError, TypeError): pass
            row=dict(key=key, signature=signature, messages=messages, seed=seed, text=raw, parsed=parsed,
                     generated_tokens=len(tokens), hit_token_limit=len(tokens)>=limit)
            with calls.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
            cache[key]=row; return parsed
        reviews=json.loads((root/'reviews.json').read_text(encoding='utf-8'))
        seed_plans=json.loads((root/'seed_plans.json').read_text(encoding='utf-8'))
        engine=Pipeline(ask, lambda key:cache[key]['text'], save, reviews, seed_plans)
        engine.run_source(source); engine.summary()
        save('status.json',dict(status='completed', calls=len(cache), review_sha256=digest(reviews)))
    except BaseException:
        if engine is not None: engine.checkpoint(); engine.summary()
        save('status.json',dict(status='failed_or_incomplete', traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__': main()

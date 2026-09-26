"""Local Hugging Face backend. Pure helpers can be tested without torch/GPU."""
import copy
import json
import os
from pathlib import Path
import re
from pipeline import VERSION, digest


def adapt_messages(messages, profile):
    messages=copy.deepcopy(messages)
    if profile!='deepseek-r1': return messages
    # R1 model card recommends putting instructions in the user message.
    system='\n\n'.join(m['content'] for m in messages if m['role']=='system')
    messages=[m for m in messages if m['role']!='system']
    if system:
        for m in messages:
            if m['role']=='user':m['content']=system+'\n\n'+m['content'];break
        else:raise ValueError('No user message for system-in-user adaptation')
    return messages


def parse_completion(text, truncated=False):
    """Score only final JSON, never JSON appearing inside a reasoning block."""
    if truncated:return None,''
    if '</think>' in text:final=text.rsplit('</think>',1)[1].strip()
    elif '<think>' in text:return None,''
    else:final=text.strip()
    if final.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',final,re.S|re.I)
        if match:final=match.group(1)
    try:
        parsed=json.loads(final)
        return (parsed,final) if isinstance(parsed,dict) else (None,final)
    except (ValueError,TypeError):return None,final


def load_sources(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    rows=json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()]
    if not isinstance(rows,list):raise ValueError('Expected JSON array or JSONL')
    seen=set()
    for s in rows:
        if not isinstance(s,dict) or not isinstance(s.get('id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]+',s['id']):
            raise ValueError('Each source needs a safe string id')
        if s['id'] in seen:raise ValueError('Duplicate source id')
        seen.add(s['id'])
        if not isinstance(s.get('question'),str) or not isinstance(s.get('answer'),str):
            raise ValueError('Each source needs string question and answer fields')
    return rows


def checkpoint_identity(path):
    path=Path(path).resolve()
    if not (path/'config.json').is_file():raise ValueError('model-path must be a local Transformers checkpoint containing config.json')
    # Record metadata, not expensive multi-GB weight hashes. Use immutable snapshots.
    files={str(p.relative_to(path)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
           for p in path.rglob('*') if p.is_file()}
    return dict(path=str(path),files=files,config_sha256=digest((path/'config.json').read_text(encoding='utf-8')))


class HFBackend:
    def __init__(self, config):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch=torch;self.config=config
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for local inference')
        self.tokenizer=AutoTokenizer.from_pretrained(config['model_path'],local_files_only=True,
                                                   trust_remote_code=config['trust_remote_code'])
        memory={int(k) if str(k).isdigit() else k:v for k,v in config['max_memory'].items()}
        kwargs=dict(local_files_only=True,device_map='auto',torch_dtype=getattr(torch,config['dtype']),
                    attn_implementation='sdpa',trust_remote_code=config['trust_remote_code'])
        if memory:kwargs['max_memory']=memory
        self.model=AutoModelForCausalLM.from_pretrained(config['model_path'],**kwargs).eval()
        mapping=getattr(self.model,'hf_device_map',{})
        if not config['allow_cpu_offload'] and any(str(v) in ('cpu','disk') for v in mapping.values()):
            raise RuntimeError('Model offloaded to CPU/disk; adjust GPU memory or explicitly allow CPU offload')
        self.runtime=dict(device_map={str(k):str(v) for k,v in mapping.items()},torch_version=torch.__version__,
                          gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                          model_class=type(self.model).__name__,tokenizer_class=type(self.tokenizer).__name__)

    def generate(self,key,messages,limit,sample):
        c=self.config;torch=self.torch;tok=self.tokenizer
        adapted=adapt_messages(messages,c['profile'])
        prompt=tok.apply_chat_template(adapted,tokenize=False,add_generation_prompt=True)
        inputs=tok(prompt,add_special_tokens=False,return_tensors='pt').to(self.model.get_input_embeddings().weight.device)
        size=inputs.input_ids.shape[1]
        if size>c['max_input_tokens']:raise ValueError('Input too long; no truncation')
        maximum=max(limit,c['max_new_tokens'])
        context=getattr(self.model.config,'max_position_embeddings',None)
        if context and size+maximum>context:raise ValueError('Input plus output budget exceeds checkpoint context length')
        seed=(c['seed']+int(digest(key)[:8],16))%(2**31)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        # R1's recommendation applies to planning/classification as well as answers.
        effective_sample=sample or c['profile']=='deepseek-r1'
        kwargs=dict(max_new_tokens=maximum,do_sample=effective_sample,
                    temperature=c['temperature'] if effective_sample else 1.,
                    top_p=c['top_p'] if effective_sample else 1.,top_k=0,
                    pad_token_id=tok.eos_token_id,use_cache=True)
        with torch.inference_mode():
            tokens=self.model.generate(**inputs,**kwargs)[0,size:]
        text=tok.decode(tokens,skip_special_tokens=True)
        # Decode without stripping special tokens as some tokenizers mark think delimiters special.
        retained=tok.decode(tokens,skip_special_tokens=False)
        if '<think>' in prompt and prompt.rfind('<think>')>prompt.rfind('</think>') and '</think>' not in retained:
            parse_text='<think>'+text  # Template opened reasoning but the model never closed it.
        elif '</think>' in retained and '</think>' not in text:
            after=retained.rsplit('</think>',1)[1]
            for special in tok.all_special_tokens:after=after.replace(special,'')
            parse_text='</think>'+after
        else:parse_text=text
        truncated=len(tokens)>=maximum
        parsed,final=parse_completion(parse_text,truncated)
        return dict(text=text,raw_with_special_tokens=retained,final_text=final,parsed=parsed,
                    seed=seed,input_tokens=size,generated_tokens=len(tokens),hit_token_limit=truncated,
                    rendered_prompt=prompt,effective_messages=adapted,effective_sampling=kwargs)


class CachedModel:
    def __init__(self,backend,output,max_calls):
        self.backend=backend;self.output=Path(output);self.max_calls=max_calls;self.cache={}
        self.path=self.output/'calls.jsonl'
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                row=json.loads(line);self.cache[row['key']]=row
    def ask(self,key,messages,limit=350,sample=True):
        signature=digest(dict(config=self.backend.config,messages=messages,limit=limit,sample=sample,protocol=VERSION))
        if key in self.cache:
            if self.cache[key]['signature']!=signature:raise ValueError('Resume cache mismatch')
            return self.cache[key]['parsed']
        if len(self.cache)>=self.max_calls:raise RuntimeError('Per-source call budget exhausted; incomplete')
        result=self.backend.generate(key,messages,limit,sample)
        row=dict(key=key,signature=signature,messages=messages,**result)
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        self.cache[key]=row;return row['parsed']
    def raw(self,key):
        row=self.cache[key]
        if self.backend.config['history']=='full':return row['text']
        # Invalid completions do not get silently replaced by invented valid answers.
        return row['final_text'] if row['parsed'] is not None else row['text']

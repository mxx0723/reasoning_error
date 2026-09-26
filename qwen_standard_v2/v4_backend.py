"""DeepSeek V4 Flash offline vLLM adapter; no HTTP or paid inference API."""
import copy
import importlib.util
from pathlib import Path
from hf_backend import parse_completion
from pipeline import digest


def load_encoder(path):
    path=Path(path)
    if not path.is_file():raise ValueError('Missing official encoding_dsv4.py; pass --encoder-path')
    spec=importlib.util.spec_from_file_location('experiment_encoding_dsv4',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in ['encode_messages','parse_message_from_completion_text','eos_token']:
        if not hasattr(module,name):raise ValueError('Official V4 encoder missing '+name)
    return module


def prepare_history(messages,history,eos):
    out=copy.deepcopy(messages)
    for message in out:
        if message['role']!='assistant':continue
        text=message['content'].removesuffix(eos)
        if '</think>' in text:
            reasoning,final=text.split('</think>',1)
            message['content']=final
            if history=='full':message['reasoning_content']=reasoning.removeprefix('<think>')
        else:message['content']=text
    return out


class V4Backend:
    def __init__(self,config):
        from vllm import LLM,SamplingParams
        import vllm
        self.config=config;self.SamplingParams=SamplingParams
        self.encoder=load_encoder(config['encoder_path'])
        kwargs=dict(config['engine_args'])
        reserved={'model','tokenizer','seed','trust_remote_code','max_model_len','generation_config'}
        if reserved.intersection(kwargs):raise ValueError('Engine args cannot override '+str(sorted(reserved.intersection(kwargs))))
        self.model=LLM(model=config['model_path'],tokenizer=config['model_path'],seed=config['seed'],
                       trust_remote_code=config['trust_remote_code'],max_model_len=config['max_model_len'],
                       generation_config='vllm',**kwargs)
        self.tokenizer=self.model.get_tokenizer()
        self.runtime=dict(vllm_version=vllm.__version__,backend='vllm_offline_v4',engine_args=kwargs,
                          encoder_path=config['encoder_path'],mode=config['thinking_mode'])

    def generate(self,key,messages,limit,sample):
        c=self.config;enc=self.encoder
        adapted=prepare_history(messages,c['history'],enc.eos_token)
        prompt=enc.encode_messages(adapted,thinking_mode=c['thinking_mode'],
                                   drop_thinking=c['history']=='final',reasoning_effort=None)
        ids=self.tokenizer.encode(prompt,add_special_tokens=False)
        maximum=max(limit,c['max_new_tokens'])
        if len(ids)>c['max_input_tokens'] or len(ids)+maximum>c['max_model_len']:
            raise ValueError('Input/output exceeds configured context; no truncation')
        seed=(c['seed']+int(digest(key)[:8],16))%(2**31)
        # Follow the V4 local deployment sampling settings for every protocol call.
        params=self.SamplingParams(temperature=c['temperature'],top_p=c['top_p'],top_k=-1,
                    max_tokens=maximum,seed=seed,skip_special_tokens=False,
                    stop=[enc.eos_token],include_stop_str_in_output=True)
        request=self.model.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)[0]
        completion=request.outputs[0];raw=completion.text
        truncated=completion.finish_reason=='length'
        parsed=None;final='';structured=None;error=None
        if completion.finish_reason=='stop':
            try:
                wire=raw if raw.endswith(enc.eos_token) else raw+enc.eos_token
                structured=enc.parse_message_from_completion_text(wire,thinking_mode=c['thinking_mode'])
                if not isinstance(structured,dict) or structured.get('tool_calls'):
                    raise ValueError('Unexpected non-answer/tool output')
                parsed,final=parse_completion(structured.get('content') or '')
            except (ValueError,TypeError,AssertionError,KeyError,IndexError) as exc:
                error=type(exc).__name__
        return dict(text=raw,raw_with_special_tokens=raw,final_text=final,parsed=parsed,
                    structured_message=structured,parse_error=error,seed=seed,
                    input_tokens=len(ids),generated_tokens=len(completion.token_ids),
                    hit_token_limit=truncated,finish_reason=completion.finish_reason,
                    rendered_prompt=prompt,effective_messages=adapted,
                    effective_sampling=dict(temperature=c['temperature'],top_p=c['top_p'],seed=seed,
                                            max_tokens=maximum,requested_sample=sample))

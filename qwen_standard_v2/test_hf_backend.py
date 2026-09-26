import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hf_backend import adapt_messages,parse_completion,CachedModel,load_sources
from local_runner import main


class FakeBackend:
    def __init__(self,config):self.config=config;self.calls=0;self.runtime={'mock':True}
    def generate(self,key,messages,limit,sample):
        self.calls+=1
        if key.endswith('/eligibility'):obj=dict(eligible=True,task_type='DIAGNOSIS')
        elif '/original_gate/' in key:obj=dict(judgment='not gold',status='single_best')
        else:obj={'judgment':'B'}
        text=json.dumps(obj)
        return dict(text='<think>private reasoning</think>'+text,final_text=text,parsed=obj)


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
    def test_system_merge_preserves_conversation(self):
        msgs=[dict(role='system',content='instruction'),dict(role='user',content='E1'),
              dict(role='assistant',content='A'),dict(role='user',content='K')]
        adapted=adapt_messages(msgs,'deepseek-r1')
        self.assertEqual([m['role'] for m in adapted],['user','assistant','user'])
        self.assertEqual(adapted[0]['content'],'instruction\n\nE1')
        self.assertEqual(msgs[0]['role'],'system')
        self.assertEqual(adapt_messages(msgs,'qwen'),msgs)
    def test_reasoning_json_is_never_scored(self):
        obj,final=parse_completion('<think>{"judgment":"A"}</think>{"judgment":"B"}')
        self.assertEqual(obj['judgment'],'B')
        self.assertNotIn('think',final)
    def test_template_opened_thinking(self):
        self.assertEqual(parse_completion('reasoning</think>{"judgment":"B"}')[0]['judgment'],'B')
    def test_unclosed_thinking_and_truncation(self):
        self.assertIsNone(parse_completion('<think>{"judgment":"A"}')[0])
        self.assertIsNone(parse_completion('{"judgment":"B"}',True)[0])
    def test_fenced_json(self):
        self.assertEqual(parse_completion('```json\n{"judgment":"B"}\n```')[0]['judgment'],'B')
        self.assertIsNone(parse_completion('some prose {"judgment":"B"}')[0])
    def test_cache_and_history(self):
        backend=FakeBackend({'history':'final'});client=CachedModel(backend,self.path,3)
        client.ask('x',[]);client.ask('x',[]);self.assertEqual(backend.calls,1)
        self.assertNotIn('think',client.raw('x'))
        restored=CachedModel(backend,self.path,3);restored.ask('x',[]);self.assertEqual(backend.calls,1)
        with self.assertRaises(ValueError):restored.ask('x',[dict(role='user',content='changed')])
        backend.config['history']='full';self.assertIn('think',client.raw('x'))
    def test_budget(self):
        client=CachedModel(FakeBackend({'history':'final'}),self.path,1);client.ask('x',[])
        with self.assertRaises(RuntimeError):client.ask('y',[])
    def test_unsafe_ids_rejected(self):
        p=self.path/'data.json';p.write_text(json.dumps([dict(id='../x',question='Q',answer='B')]))
        with self.assertRaises(ValueError):load_sources(p)
    def fixture(self):
        model=self.path/'model';model.mkdir();(model/'config.json').write_text('{}')
        data=self.path/'data.json';data.write_text(json.dumps([dict(id='one',question='Question?',answer='B')]))
        return ['--profile','deepseek-r1','--model-path',str(model),'--data',str(data),'--output',str(self.path/'run')]
    def test_dry_run_without_gpu_import_or_output(self):
        args=self.fixture()
        with patch('local_runner.HFBackend',side_effect=AssertionError('must not load model')):
            with contextlib.redirect_stdout(io.StringIO()):main(args+['--dry-run'])
        self.assertFalse((self.path/'run').exists())
    def test_full_entry_point_gate_and_resume(self):
        args=self.fixture()
        with patch('local_runner.HFBackend',FakeBackend),contextlib.redirect_stdout(io.StringIO()):
            main(args);main(args)
        out=self.path/'run'/'one'
        self.assertEqual(len((out/'calls.jsonl').read_text().splitlines()),4)
        audit=json.loads((out/'audit.json').read_text())
        self.assertEqual(audit[0]['stage'],'original_gate_not_passed_pending_semantic_review')
        self.assertFalse((self.path/'run'/'active.lock').exists())


if __name__=='__main__':unittest.main()

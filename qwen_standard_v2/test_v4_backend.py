import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from v4_backend import V4Backend,load_encoder,prepare_history


class Encoder:
    eos_token='<EOS>'
    def encode_messages(self,messages,**kwargs):
        self.messages=messages;self.options=kwargs;return 'ENCODED'
    def parse_message_from_completion_text(self,text,thinking_mode):
        if thinking_mode=='thinking':
            reasoning,text=text.split('</think>',1)
        else:reasoning=''
        return dict(content=text.removesuffix(self.eos_token),reasoning_content=reasoning,tool_calls=[])


class Tests(unittest.TestCase):
    def backend(self,mode='chat',finish='stop',raw='{"judgment":"B"}<EOS>'):
        b=V4Backend.__new__(V4Backend)
        b.encoder=Encoder();b.SamplingParams=lambda **kw:kw
        b.config=dict(history='final',thinking_mode=mode,max_new_tokens=10,max_input_tokens=30,
                      max_model_len=100,seed=1,temperature=1.,top_p=1.)
        b.tokenizer=SimpleNamespace(encode=lambda text,**kw:[1,2,3])
        def generate(inputs,params,**kw):
            b.sent=(inputs,params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=raw,finish_reason=finish,token_ids=[4,5])])]
        b.model=SimpleNamespace(generate=generate)
        return b
    def test_offline_tokens_and_final_score(self):
        b=self.backend();r=b.generate('x',[dict(role='user',content='Q')],5,True)
        self.assertEqual(r['parsed']['judgment'],'B')
        self.assertEqual(b.sent[0],[{'prompt_token_ids':[1,2,3]}])
        self.assertTrue(b.encoder.options['drop_thinking'])
    def test_thinking_final_only(self):
        b=self.backend('thinking',raw='{"judgment":"A"}</think>{"judgment":"B"}<EOS>')
        self.assertEqual(b.generate('x',[],5,True)['parsed']['judgment'],'B')
    def test_truncation_rejected(self):
        self.assertIsNone(self.backend(finish='length').generate('x',[],5,True)['parsed'])
    def test_history_has_separate_reasoning_field(self):
        msgs=[dict(role='assistant',content='thought</think>{"judgment":"A"}<EOS>')]
        self.assertEqual(prepare_history(msgs,'full','<EOS>')[0]['reasoning_content'],'thought')
        self.assertNotIn('reasoning_content',prepare_history(msgs,'final','<EOS>')[0])
    def test_missing_encoder_fails(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):load_encoder(Path(d)/'missing.py')
    @unittest.skipUnless(os.getenv('DEEPSEEK_V4_ENCODER'),'Set DEEPSEEK_V4_ENCODER for official encoder integration')
    def test_official_encoder_chat_and_thinking(self):
        encoder=load_encoder(os.environ['DEEPSEEK_V4_ENCODER'])
        for mode,raw in [('chat','{"judgment":"B"}'),('thinking','thought</think>{"judgment":"B"}')]:
            b=self.backend(mode,raw=raw+encoder.eos_token);b.encoder=encoder
            r=b.generate('x',[dict(role='system',content='S'),dict(role='user',content='Q')],5,True)
            self.assertEqual(r['parsed']['judgment'],'B')
            self.assertTrue(r['rendered_prompt'].endswith('<think>' if mode=='thinking' else '</think>'))
        msgs=prepare_history([dict(role='user',content='E1'),dict(role='assistant',content='thought</think>A'+encoder.eos_token),dict(role='user',content='K')],'full',encoder.eos_token)
        kept=encoder.encode_messages(msgs,thinking_mode='thinking',drop_thinking=False)
        dropped=encoder.encode_messages(msgs,thinking_mode='thinking',drop_thinking=True)
        self.assertIn('thought',kept);self.assertNotIn('thought',dropped)


if __name__=='__main__':unittest.main()

import copy
import json
import unittest
from pipeline import Pipeline, build_case, digest, messages

SOURCE=dict(id='test',question='A patient has several early nonspecific symptoms. New laboratory evidence supports another decision. What is the most likely diagnosis?',answer='B')
PLAN=dict(type='UPDATE',task_type='DIAGNOSIS',atom_spans=[['New laboratory evidence supports another decision.']])
def ans(text, status='single_best'): return dict(judgment=text,status=status,alternatives=[])

class Fake:
    def __init__(self, kind='UPDATE', fail_original=False, fail_final=False):
        self.keys=[]; self.store={}; self.kind=kind; self.fail_original=fail_original; self.fail_final=fail_final
    def ask(self,key,msgs,**kwargs):
        self.keys.append(key)
        if key.endswith('/eligibility'): value=dict(eligible=True,task_type='DIAGNOSIS')
        elif key.endswith('/construct'): value=dict(proposals=[dict(PLAN,type=self.kind)])
        elif '/original_gate/' in key: value=ans('Wrong' if self.fail_original else 'B')
        elif self.fail_final and '/final_holdout/Fresh/' in key: value=ans('Wrong')
        elif '/E1/' in key: value=ans('A' if self.kind=='UPDATE' else 'B')
        else: value=ans('B')
        self.store[key]=dict(messages=msgs,answer=value); return value
    def raw(self,key): return json.dumps(self.store[key]['answer'])

class Tests(unittest.TestCase):
    def engine(self, fake, reviews=None):
        files={}; p=Pipeline(fake.ask,fake.raw,lambda k,v:files.update({k:copy.deepcopy(v)}),reviews)
        return p,files
    def test_original_gate_prevents_construction(self):
        f=Fake(fail_original=True); p,_=self.engine(f); p.run_source(SOURCE)
        self.assertFalse(any('/construct' in k or '/dev/' in k for k in f.keys))
    def test_no_substring_false_positive_and_review_rescue(self):
        f=Fake(); p,_=self.engine(f)
        s=dict(SOURCE,answer='CLL')
        self.assertNotEqual(p.label(s,ans('CML')), 'B')
        self.assertNotEqual(p.label(s,ans('not CLL')), 'B')
        r={'sources':{digest(s):dict(reviewed=True,gold_aliases=['chronic lymphocytic leukemia'])}}
        p,_=self.engine(f,r)
        self.assertEqual(p.label(s,ans('chronic lymphocytic leukemia','differential')),'B')
    def test_exact_source_spans(self):
        self.assertNotIn(PLAN['atom_spans'][0][0], build_case(SOURCE,PLAN,0)['E1'])
        with self.assertRaises(ValueError): build_case(SOURCE,dict(PLAN,atom_spans=[['invented laboratory finding']]),0)
        with self.assertRaises(ValueError): build_case(SOURCE,dict(PLAN,atom_spans=PLAN['atom_spans']*2),0)
    def test_update_order_and_independent_holdout(self):
        f=Fake(); p,files=self.engine(f); p.run_source(SOURCE)
        self.assertLess(f.keys.index('test/original_gate/2'),f.keys.index('test/construct'))
        self.assertTrue(any('/minimal_validation/' in k for k in f.keys))
        self.assertEqual(len(files['behavior_trials.json']),15)
        rec=p.audit[0]['cases'][0]
        self.assertTrue(rec['behavior_validated_candidate']); self.assertFalse(rec['formal_eligible'])
        fresh=f.store['test_v2_0/final_holdout/Fresh/0']['messages']
        prior=f.store['test_v2_0/behavior/Fixed_prior_label/0']['messages']
        self.assertEqual(prior[-1]['content'],fresh[-1]['content']+'\n\nRecorded prior assessment: A.')
        seq=f.store['test_v2_0/behavior/Sequential/0']['messages']
        self.assertEqual(seq[-2]['content'],f.raw('test_v2_0/final_holdout/E1/0'))
    def test_failed_holdout_never_enters_behavior(self):
        f=Fake(fail_final=True); p,_=self.engine(f); p.run_source(SOURCE)
        self.assertFalse(any('/behavior/' in k for k in f.keys))
        self.assertTrue(p.audit[0]['cases'][0]['full_pair_stable'])
    def test_maintain_not_called_revision_or_minimal_search(self):
        f=Fake(kind='MAINTAIN'); p,files=self.engine(f); p.run_source(SOURCE)
        self.assertFalse(any('/minimal_validation/' in k for k in f.keys))
        self.assertEqual({t['outcome'] for t in files['behavior_trials.json']},{'maintain'})
    def test_no_gold_or_keypoints_in_answer_prompt(self):
        f=Fake(); p,_=self.engine(f); s=dict(SOURCE,answer_key_points=['SECRET_METADATA'])
        p.run_source(s)
        for k,v in f.store.items():
            if not k.endswith('/construct'): self.assertNotIn('SECRET_METADATA',json.dumps(v['messages']))
    def test_seed_plans_cannot_bypass_original_gate(self):
        f=Fake(fail_original=True); p,_=self.engine(f); p.seed_plans={'test':[PLAN]}
        p.run_source(SOURCE)
        self.assertNotIn('cases',p.audit[0])
    def test_review_releases_original_gate(self):
        f=Fake(fail_original=True)
        r={'sources':{digest(SOURCE):dict(reviewed=True,gold_aliases=['Wrong'])}}
        p,_=self.engine(f,r); p.run_source(SOURCE)
        self.assertIn('test/construct',f.keys)
    def test_minimal_search_checks_all_smaller_groups(self):
        source=dict(SOURCE,question=SOURCE['question']+' Another independently observed finding was recorded.')
        plan=dict(PLAN,atom_spans=PLAN['atom_spans']+[['Another independently observed finding was recorded.']])
        f=Fake(); original=f.ask
        def ask(key,msgs,**kw):
            value=original(key,msgs,**kw)
            if '/minimal_validation/0/' in key: value=ans('A')
            f.store[key]['answer']=value
            return value
        p=Pipeline(ask,f.raw,lambda k,v:None,seed_plans={'test':[plan]}); p.run_source(source)
        self.assertEqual(p.audit[0]['cases'][0]['selected_K_indices'],[1])
        self.assertEqual(len([k for k in f.keys if '/minimal_validation/' in k]),10)

if __name__=='__main__': unittest.main()

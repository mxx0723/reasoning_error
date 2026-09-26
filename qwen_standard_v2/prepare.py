"""Prepare a new immutable-protocol workspace without changing historical runs."""
import json
from pathlib import Path
from pipeline import build_case, norm

root=Path(__file__).resolve().parent
old=root.parent/'qwen_full_scope_v1_20260924'
rows=[json.loads(line) for line in (old/'oeq.jsonl').read_text(encoding='utf-8').splitlines()]
def write(name, obj):
    (root/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),encoding='utf-8')
sources=[s for s in rows if isinstance(s.get('answer'),str) and s['answer'].strip()
         and isinstance(s.get('question'),str) and s['question'].strip()]
sources.sort(key=lambda s:(s.get('source') in ['pubmedqa','liveqa'],s['id']))
assert len({s['id'] for s in sources})==len(sources)
write('sources.json',sources)
write('coverage_manifest.json',[dict(source_id=s['id'],status='original_gate_required' if s in sources
                                   else 'missing_question_or_gold') for s in rows])
config=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
write('protocol.json',dict(protocol_version='standard_v2',model=config['model'],model_path=config['model_path'],
      temperature=config['temperature'],max_input_tokens=config['max_input_tokens'],seed=20260926,
      max_calls_per_source=400,original_gate='2/3',development='2/3',pair_validation='4/5',
      minimal_search='exhaustive cardinality order within <=4 prespecified atom groups, 4/5',
      independent_final_holdout='4/5',behavior_repeats=5,
      clinical_approval_required_for_formal_export=True))
if not (root/'reviews.json').exists(): write('reviews.json',dict(sources={},answers={},cases={}))
legacy=[]
for name in ['qwen_atomic_pairs_v1_20260923','qwen_manual_expand_v2_20260924','qwen_manual_expand_v3_20260924','qwen_full_scope_v1_20260924']:
    p=root.parent/name
    for f in p.glob('**/*audit.json'):
        legacy.append(dict(path=str(f),standard_v2_status='legacy_evidence_only_not_certified'))
write('legacy_inventory.json',legacy)
# Reuse exact existing manual evidence partitions, but NEVER their pass/fail labels.
seed_plans={}; migration=[]; by_id={s['id']:s for s in sources}
for name in ['qwen_manual_expand_v2_20260924','qwen_manual_expand_v3_20260924','qwen_atomic_pairs_v1_20260923']:
    p=root.parent/name
    file=p/('cases.json' if 'atomic' in name else 'batches.json')
    data=json.loads(file.read_text(encoding='utf-8'))
    cases=data if 'atomic' in name else [c for batch in data for c in batch]
    for c in cases:
        sid=c['source_id']; rec=dict(case_id=c['case_id'],source_id=sid)
        try:
            source=by_id[sid]
            if c.get('atoms'):
                groups=[[a['text']] for a in c['atoms']]
            else:
                spans=c.get('source_spans',[])
                groups=[[s['text']] for s in spans]
            plan=dict(type=c.get('type','UPDATE'),task_type=c['task_type'],atom_spans=groups,
                      legacy_case_id=c['case_id'],why_K_matters=c.get('rationale','Legacy manual partition'))
            rebuilt=build_case(source,plan,0)
            if norm(rebuilt['E1'])!=norm(c['E1']): raise ValueError('Legacy E1 is not exact deletion of raw source; requires review')
            seed_plans.setdefault(sid,[]).append(plan); rec['status']='reuse_stimulus_only_all_gates_required'
        except (ValueError,TypeError,KeyError) as exc:
            rec.update(status='legacy_partition_pending_review',reason=str(exc))
        migration.append(rec)
write('seed_plans.json',seed_plans); write('legacy_migration.json',migration)
sb='''#!/bin/bash
#SBATCH --job-name=qwen_standard_v2
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --exclude=kitt
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
set -euo pipefail
export PYTHONUTF8=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
cd "ROOT"
exec /nethome/xma394/miniconda3/envs/moma/bin/python runner.py
'''.replace('ROOT',str(root))
(root/'run.sbatch').write_text(sb,encoding='utf-8')
print(json.dumps(dict(total=len(rows),sources=len(sources),missing=len(rows)-len(sources),submitted=False)))

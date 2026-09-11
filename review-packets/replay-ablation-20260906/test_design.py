import unittest
from collections import Counter
from shared import *
from audit_and_bon import exact_bon
from selfsight.v4.factual_truth import factual_answer


class DesignTests(unittest.TestCase):
    def test_n_without_replacement(self):
        candidates=[{'candidate_id':'a','correct':True,'sampling_seed':1},{'candidate_id':'b','correct':False,'sampling_seed':0}]
        scores={'a':1,'b':1}
        self.assertEqual(exact_bon(candidates,scores,1)['uniform_top'],.5)
        self.assertEqual(exact_bon(candidates,scores,2)['uniform_top'],.5)
        self.assertEqual(exact_bon(candidates,scores,2)['deterministic'],0)
        self.assertEqual(exact_bon(candidates,scores,2)['oracle_any_correct'],1)
        with self.assertRaises(ValueError):exact_bon(candidates,scores,8)

    def test_replay_controls_preserve_slot_budget(self):
        schedule=read(SIDE/'schedule.json');all_orig=[r for rd in schedule.values() for r in rd['original']]
        excluded_ids={r['sample_id'] for r in all_orig};excluded_images={r['image_path'] for r in all_orig}
        for rd in schedule.values():
            for key in ('original','balanced_absence','alternate_naive','alternate_rfo_gold'):self.assertEqual(len(rd[key]),16)
            for arm in ('naive','rfo_gold'):
                for a,b in zip(rd['original'],rd['alternate_'+arm]):
                    self.assertEqual(a['answer'],b['answer']);self.assertNotIn(b['sample_id'],excluded_ids);self.assertNotIn(b['image_path'],excluded_images)
            for a,b in zip(rd['original'],rd['balanced_absence']):self.assertEqual(a['image_path'],b['image_path'])
        changed=Counter(r['answer'] for rd in schedule.values() for r in rd['balanced_absence'])
        self.assertEqual(changed['no'],15);self.assertEqual(changed['0'],16)

    def test_training_absence_ground_truth(self):
        split=read(MAIN/'split.json');verdicts={str(Path(r['image_path']).resolve()):r for run in split['runs'] for r in lines(ROOT/run/'verified.jsonl')}
        for rd in read(SIDE/'schedule.json').values():
            for item in rd['balanced_absence']:
                if 'fact_provenance' not in item:continue
                v=verdicts[str(Path(item['image_path']).resolve())];self.assertTrue(v['image_correct'])
                fact=factual_answer(item['fact_provenance']['question'],v)
                self.assertTrue(fact.known);self.assertEqual(fact.answer,item['answer'])

    def test_probe_balance_and_holdout(self):
        probe=read(SIDE/'probe.json')['rows'];self.assertEqual(len(probe),100)
        self.assertEqual(Counter(r['group'] for r in probe),Counter({'count_zero':25,'count_positive':25,'existence_no':25,'existence_yes':25}))
        primary={p['scene_sha256'] for p in read(OLD/'fixed-bank.json')['pools'] if p['primary_scene_disjoint']}
        self.assertEqual({r['scene_sha256'] for r in probe},primary)
        self.assertEqual(len({r['id'] for r in probe}),100)

    def test_constant_no_is_not_a_repair(self):
        rows=[]
        for condition in ('prompt_on','prompt_off'):
            for group,truth in [('existence_no','no'),('existence_yes','yes'),('count_zero','0'),('count_positive','1')]:
                rows.append({'condition':condition,'group':group,'truth':truth,'answer':{'normalized_answer':'no','abstain':False,'error':None}})
        summary=summarize_probe(rows)
        self.assertEqual(summary['prompt_on']['existence_no']['accuracy'],1)
        self.assertEqual(summary['prompt_on']['existence_yes']['accuracy'],0)
        self.assertEqual(summary['prompt_on']['count_zero']['accuracy'],0)


if __name__=='__main__':unittest.main()

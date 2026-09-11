import unittest
from common import *
from prepare import fresh_specs
from analyze_a import interval,keyed_loss

class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=[r for name in ('2plus1','1plus1plus1') for r in lines(ROOT/f'data/v4/corpus_{name}.jsonl')]
        cls.rows,cls.vocab,cls.existing=fresh_specs(cls.old)

    def test_new_scene_sets_disjoint_and_balanced(self):
        self.assertEqual(len(self.rows),166)
        sets=[tuple(r['noun_set']) for r in self.rows]
        self.assertEqual(len(set(sets)),166);self.assertFalse(set(sets)&self.existing)
        for split,n in [('A',8),('B',75)]:
            for size in (2,3):self.assertEqual(sum(r['split']==split and len(r['spec']['objects'])==size for r in self.rows),n)

    def test_negative_atoms_are_unique_and_spec_absent(self):
        for r in self.rows:
            q=questions(r['spec'],self.vocab);extra=q['questions'][q['n_original']:]
            self.assertEqual(len(q['questions']),4*len(r['spec']['objects']))
            self.assertEqual(len(set(x['text'] for x in q['questions'])),len(q['questions']))
            for item in extra:
                noun=item['atom_id'].split(':')[-1]
                self.assertNotIn(noun,r['noun_set']);self.assertIn(item['expected_answer'],('no','0'))

    def test_sampler_deterministic_without_answers(self):
        rows,vocab,existing=fresh_specs(self.old)
        self.assertEqual(rows,self.rows);self.assertEqual(vocab,self.vocab)

    def test_generation_seeds_paired_and_unique(self):
        keys=[seed('generation',r['spec']['spec_id'],i) for r in self.rows for i in range(6)]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertNotEqual(seed('loss','training','x',0),seed('loss','training','x',1))

    def test_loss_pairing_refuses_missing_or_duplicate_rows(self):
        rows=[{'cohort':'a','scene_key':str(i//2),'noise_index':i%2} for i in range(64)]
        self.assertEqual(len(keyed_loss(rows)),64)
        with self.assertRaises(AssertionError):keyed_loss(rows[:-1])
        with self.assertRaises(AssertionError):keyed_loss(rows[:-1]+[rows[0]])

    def test_interval_reports_equal_and_negative_responses(self):
        z=interval([0.]*16);self.assertEqual(z['ci95'],[0.,0.])
        d=interval([-.02]*16);self.assertLess(d['ci95'][1],0)

if __name__=='__main__':unittest.main()

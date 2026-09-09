"""Bounded tests for question scope, unknown facts and tied-score semantics."""
import unittest
from common import *
from selfsight.v4.probe import spec_questions


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.spec = {'spec_id': 'test-canonical', 'prompt': 'one red cup, one blue notebook',
                     'objects': [{'object':'cup','color':'red','count':1}, {'object':'notebook','color':'blue','count':1}]}
        self.vocab = ['cup','mug','book','notebook','apple','pear','bowl','bottle']

    def test_canonical_exclusion_and_determinism(self):
        a = neg_questions(self.spec, self.vocab, 2)
        b = neg_questions(self.spec, list(reversed(self.vocab)), 2)
        self.assertEqual(a,b)
        nouns = [q.atom_id.split(':')[-1] for q in a]
        self.assertEqual(len(set(nouns)),2)
        self.assertFalse(set(nouns) & {'mug','book'})

    def test_old_questions_unchanged(self):
        spec=SceneSpec.from_dict(self.spec)
        before=spec_questions(spec)
        neg_questions(self.spec,self.vocab,2)
        self.assertEqual(before,spec_questions(spec))

    def test_correct_image_observed_perfectly_scores_full(self):
        spec=SceneSpec.from_dict(self.spec)
        qs=list(spec_questions(spec))+neg_questions(self.spec,self.vocab,2)
        v={'resolution':'human','detections':[{'object':'mug','color':'red'},{'object':'book','color':'blue'}]}
        self.assertTrue(all(factual_answer(as_serializable(q),v).answer==q.expected_answer for q in qs))

    def test_extra_present_is_yes_not_gold_no(self):
        q=neg_questions(self.spec,self.vocab,1)[0]
        noun=q.atom_id.split(':')[-1]
        f=factual_answer(as_serializable(q),{'resolution':'human','detections':[{'object':noun,'color':'red'}]})
        self.assertTrue(f.known)
        self.assertEqual((f.answer,q.expected_answer),('yes','no'))

    def test_disputed_absence_remains_unknown(self):
        q=neg_questions(self.spec,self.vocab,1)[0]
        noun=q.atom_id.split(':')[-1]
        v={'resolution':'agreed_verdict','detections':[], 'disputed':[{'object':noun,'color':'red'}]}
        self.assertFalse(factual_answer(as_serializable(q),v).known)

    def test_tie_mean_not_actual_ceiling(self):
        cs=[{'candidate_id':'a','sampling_seed':0,'correct':True}, {'candidate_id':'b','sampling_seed':1,'correct':False}]
        m=pool_metrics(cs,{'a':1,'b':1})
        self.assertEqual(m['top_correct_rate'],.5)
        self.assertTrue(m['selected_correct'])

    def test_spec_recitation_still_gets_every_mark(self):
        qs=list(spec_questions(SceneSpec.from_dict(self.spec)))+neg_questions(self.spec,self.vocab,2)
        answers=[q.expected_answer for q in qs]
        self.assertEqual(sum(a==q.expected_answer for a,q in zip(answers,qs))/len(qs),1)
        self.assertIn('no', answers)


if __name__ == '__main__':
    unittest.main()

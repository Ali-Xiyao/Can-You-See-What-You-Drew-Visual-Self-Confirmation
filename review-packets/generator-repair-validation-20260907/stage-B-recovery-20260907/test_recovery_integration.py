"""Scientific boundaries that must survive parse-failure quarantine."""
import pytest
import runtime_common as runtime
from selfsight.v4.factual_truth import factual_answer
from bcommon import image_gold


def test_quarantine_is_unknown_for_both_endpoints():
    v = {'resolution': 'pending_human', 'image_correct': None, 'detections': [],
         'factual_truth_unavailable': True}
    q = {'family': 'count', 'atom_id': 's:count:0',
         'text': 'How many red roses are in this picture? Answer with a single whole number only.'}
    assert image_gold(v) is None
    answer = factual_answer(q, v)
    assert not answer.known and answer.answer is None


def test_only_user_approved_deadline_is_accepted(monkeypatch):
    monkeypatch.setattr(runtime, 'main_paused', lambda: None)
    monkeypatch.setattr(runtime, 'approved_deadline', lambda: 200)
    monkeypatch.setattr(runtime.time, 'time', lambda: 100)
    class NoFailure:
        def __truediv__(self, name): return self
        def exists(self): return False
    monkeypatch.setattr(runtime, 'R', NoFailure())
    runtime.check_time(200)
    with pytest.raises(AssertionError, match='Unapproved'):
        runtime.check_time(201)
    with pytest.raises(AssertionError, match='Unapproved'):
        runtime.check_time(199)
    monkeypatch.setattr(runtime.time, 'time', lambda: 200)
    with pytest.raises(AssertionError, match='exhausted'):
        runtime.check_time(200)


def test_unknown_label_preserves_selection_worst_completion():
    from stats import paired_selection
    # The failed image cannot be silently classified wrong or excluded.
    assert paired_selection([1, 0], [0, 1], [None, 1]) == pytest.approx([-1, 0])
    assert paired_selection([1, 0], [1, 0], [None, 1]) == pytest.approx([0, 0])

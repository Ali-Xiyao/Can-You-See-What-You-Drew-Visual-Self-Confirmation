"""Synthetic tests only; no live model scores or partial gold are inspected."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('secondary_analysis', Path(__file__).with_name('analyze_secondary.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

MODELS = ['base', 'original', 'no_zero', 'zero_gradient']
CONDITIONS = ['prompt_on', 'prompt_blank', 'prompt_off']


def records(pools):
    return [{'model': model, 'condition': condition, 'spec_id': sid,
             'labels': labels.copy(), 'original_scores': [1, 0, 0, 0, 0, 0] if model == 'base' else [0, 1, 0, 0, 0, 0]}
            for model in MODELS for condition in CONDITIONS for sid, labels in pools.items()]


def test_pending_tentative_true_never_becomes_known():
    assert analysis.frozen.image_gold({'resolution': 'pending_human', 'image_correct': True}) is None
    assert analysis.frozen.image_gold({'resolution': 'agreed', 'image_correct': True}) == 1


@pytest.mark.parametrize('labels,availability,structure', [
    ([0]*6, 'certifiedallwrong', 'all_wrong'),
    ([1]*6, 'atleast1knowntrue', 'all_correct'),
    ([1, 0, None, None, None, None], 'atleast1knowntrue', 'confirmed_mixed'),
    ([0, 0, None, None, None, None], 'unresolvednoneknowntrue', 'unresolved_membership'),
    ([1, 1, None, None, None, None], 'atleast1knowntrue', 'unresolved_membership'),
    ([None]*6, 'unresolvednoneknowntrue', 'unresolved_membership'),
])
def test_all_groups(labels, availability, structure):
    item = analysis.classify(labels)
    assert item['availability'] == availability and item['structure'] == structure
    assert item['c'] + item['f'] + item['u'] == 6


def test_all_true_and_false_pools_have_zero_paired_selection_change():
    pools = {'wrong': [0]*6, 'correct': [1]*6}
    report = analysis.summarize(records(pools), MODELS, CONDITIONS, pools)
    for group in ('all_wrong', 'all_correct'):
        for result in report['strata'][group]['paired_changes_vs_base'].values():
            assert result['mean_bounds'] == [0, 0]
            assert result['ci90_worst_completion'] == result['ci95_worst_completion'] == [0, 0]


def test_shared_unknown_labels_cancel_in_paired_difference():
    pools = {'unknown': [None]*6}
    items = records(pools)
    for row in items:
        row['original_scores'] = [1, 0, 0, 0, 0, 0]
    report = analysis.summarize(items, MODELS, CONDITIONS, pools)
    group = report['strata']['unresolved_membership']
    assert group['summaries']['base:prompt_on']['top_correct_rate']['mean_bounds'] == [0, 1]
    assert all(value['mean_bounds'] == [0, 0] for value in group['paired_changes_vs_base'].values())


def test_every_model_and_condition_uses_same_stratum_denominator():
    pools = {'mixed1': [1, 0, 0, 0, 0, 0], 'mixed2': [1, 0, None, None, None, None],
             'unknown': [None]*6, 'wrong': [0]*6}
    report = analysis.summarize(records(pools), MODELS, CONDITIONS, pools)
    for entry in report['strata'].values():
        if not entry['estimated']:
            continue
        assert len(entry['summaries']) == 12 and len(entry['paired_changes_vs_base']) == 9
        assert all(result['top_correct_rate']['n_scenes'] == entry['n_scenes'] for result in entry['summaries'].values())
        assert all(result['n_scenes'] == entry['n_scenes'] for result in entry['paired_changes_vs_base'].values())


def test_inconsistent_shared_labels_are_rejected():
    pools = {'mixed': [1, 0, 0, 0, 0, 0]}
    items = records(pools)
    items[-1]['labels'][0] = 0
    with pytest.raises(AssertionError, match='shared image gold'):
        analysis.validate_per_pool(items, MODELS, CONDITIONS, pools)


def test_empty_stratum_is_not_estimated():
    pools = {'wrong': [0]*6}
    report = analysis.summarize(records(pools), MODELS, CONDITIONS, pools)
    group = report['strata']['confirmed_mixed']
    assert group['n_scenes'] == 0 and group['estimated'] is False
    assert group['summaries'] == group['paired_changes_vs_base'] == {}


@pytest.mark.parametrize('mutator', [lambda rows: rows[:-1], lambda rows: rows + [rows[0]]])
def test_missing_or_repeated_pool_is_rejected(mutator):
    pools = {'wrong': [0]*6}
    with pytest.raises(AssertionError):
        analysis.validate_per_pool(mutator(records(pools)), MODELS, CONDITIONS, pools)


def test_confirmed_mixed_retains_unknown_selected_candidate_bounds():
    pools = {'mixed_with_unknown': [1, 0, None, None, None, None]}
    items = records(pools)
    for row in items:
        if row['model'] != 'base':
            row['original_scores'] = [0, 0, 1, 0, 0, 0]
    report = analysis.summarize(items, MODELS, CONDITIONS, pools)
    group = report['strata']['confirmed_mixed']
    assert group['n_scenes'] == 1 and report['strata']['unresolved_membership']['n_scenes'] == 0
    assert all(result['mean_bounds'] == [-1, 0] for result in group['paired_changes_vs_base'].values())
    assert all(result['ci95_worst_completion'] == [-1, 0] for result in group['paired_changes_vs_base'].values())


def fake_registration(tmp_path, monkeypatch):
    secondary, stage = tmp_path / 'secondary', tmp_path / 'stage'
    secondary.mkdir(); stage.mkdir()
    monkeypatch.setattr(analysis, 'S', secondary)
    monkeypatch.setattr(analysis, 'B', stage)
    sources = [secondary/'analyze_secondary.py', secondary/'test_secondary.py', stage/'stats.py', stage/'bcommon.py']
    for path in sources:
        path.write_text('frozen test source', encoding='utf-8')
    registration = secondary / 'REGISTRATION.json'
    registration.write_text(json.dumps({'timestamp': '2000-01-01T00:00:00+00:00',
        'prior_data_seen': {'completed_batches': ['batch-00'], 'model_selection_comparisons_seen': False},
        'source_sha256': {str(stage/'stats.py'): analysis.sha(stage/'stats.py')}}), encoding='utf-8')
    freeze = {'timestamp': '2000-01-02T00:00:00+00:00',
              'registration_sha256': analysis.sha(registration),
              'source_sha256': {str(p): analysis.sha(p) for p in sources}}
    return registration, secondary/'CODE_FREEZE.json', freeze, sources


def test_separate_code_freeze_binds_to_immutable_registration(tmp_path, monkeypatch):
    registration, path, freeze, _ = fake_registration(tmp_path, monkeypatch)
    original = registration.read_bytes()
    path.write_text(json.dumps(freeze), encoding='utf-8')
    result = analysis.registration(require_source_freeze=True)
    assert result['code_freeze']['sha256'] == analysis.sha(path)
    assert registration.read_bytes() == original


def test_completed_results_gate_requires_separate_code_freeze(tmp_path, monkeypatch):
    fake_registration(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        analysis.registration(require_source_freeze=True)


@pytest.mark.parametrize('change', ['registration_binding', 'source_bytes', 'freeze_timestamp'])
def test_code_freeze_rejects_changed_binding_or_code(tmp_path, monkeypatch, change):
    _, path, freeze, sources = fake_registration(tmp_path, monkeypatch)
    if change == 'registration_binding':
        freeze['registration_sha256'] = 'wrong'
    elif change == 'source_bytes':
        sources[0].write_text('changed code', encoding='utf-8')
    else:
        freeze['timestamp'] = '1999-01-01T00:00:00+00:00'
    path.write_text(json.dumps(freeze), encoding='utf-8')
    with pytest.raises(AssertionError):
        analysis.registration(require_source_freeze=True)

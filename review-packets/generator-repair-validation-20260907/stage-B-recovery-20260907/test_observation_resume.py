import copy
import json
import pytest
import observation_resume as resume


@pytest.fixture
def sample():
    question = {'question_id': 's:count', 'atom_id': 's:count', 'family': 'count',
        'text': 'How many red books are in this picture?', 'expected_answer': '2'}
    scene = {'scene_sha256': 'scenehash', 'spec': {'prompt': 'two red books'},
        'n_original': 1, 'questions': [question]}
    image = {'spec_id': 's', 'scene_sha256': 'scenehash', 'candidate_index': 0,
        'seed': 13, 'image_path': 'image.png', 'rgb_sha256': 'rgbhash', 'prompt': 'two red books'}
    p = {'observer_revision': 'rev', 'conditions': ['prompt_on', 'prompt_blank', 'prompt_off']}
    model = {'id': 'base', 'parameter_digest': 'parameters'}
    row = {'observer': 'base', 'generator': 'base', 'spec_id': 's', 'scene_sha256': 'scenehash',
        'candidate_index': 0, 'seed': 13, 'condition': 'prompt_on', 'n_original': 1,
        'image_path': 'image.png', 'device': 'cuda:0', 'observation': {
            'request_id': 'request', 'observer_id': 'showlab/show-o2-1.5B', 'observer_revision': 'rev',
            'rgb_sha256': 'rgbhash', 'answers': [{'question_id': 's:count', 'raw_answer': '2',
                'normalized_answer': '2', 'abstain': False, 'error': None, 'latency_ms': 10}],
            'started_at': 'old', 'finished_at': 'old'}}
    return scene, image, p, model, row


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def test_condition_boundary_prefix_is_exact_and_can_end_mid_image(tmp_path, sample):
    scene, image, p, model, row = sample
    second = copy.deepcopy(row); second['condition'] = 'prompt_blank'
    path = tmp_path / 'answers.jsonl'; write_rows(path, [row, second])
    expected = resume.expected_sequence([image], p['conditions'])
    rows, n = resume.validate_prefix(path, expected, {'s': scene}, p, model, 0)
    assert len(rows) == n == 2
    assert expected[len(rows)][1] == 'prompt_off'
    # An equal-length set of rows in the wrong order is not a usable prefix.
    write_rows(path, [second, row])
    with pytest.raises(AssertionError, match='identity mismatch'):
        resume.validate_prefix(path, expected, {'s': scene}, p, model, 0)


@pytest.mark.parametrize('change', ['image', 'scene', 'device', 'rgb', 'question', 'raw', 'error', 'partial', 'abstain'])
def test_corrupt_prefix_is_never_silently_reused(tmp_path, sample, change):
    scene, image, p, model, row = sample
    if change == 'image': row['image_path'] = 'other.png'
    elif change == 'scene': row['scene_sha256'] = 'other'
    elif change == 'device': row['device'] = 'cuda:1'
    elif change == 'rgb': row['observation']['rgb_sha256'] = 'other'
    elif change == 'question': row['observation']['answers'][0]['question_id'] = 'other'
    elif change == 'raw': row['observation']['answers'][0]['raw_answer'] = '3'
    elif change == 'error': row['observation']['answers'][0]['error'] = 'failure'
    elif change == 'partial': row['observation']['answers'] = []
    elif change == 'abstain': row['observation']['answers'][0]['abstain'] = True
    path = tmp_path / 'answers.jsonl'; write_rows(path, [row])
    with pytest.raises(AssertionError):
        resume.validate_prefix(path, resume.expected_sequence([image], p['conditions']), {'s': scene}, p, model, 0)


def test_incomplete_tail_is_not_discarded(tmp_path, sample):
    scene, image, p, model, row = sample
    path = tmp_path / 'answers.jsonl'
    path.write_text(json.dumps(row), encoding='utf-8')
    with pytest.raises(AssertionError, match='Incomplete JSONL'):
        resume.validate_prefix(path, [(image, 'prompt_on')], {'s': scene}, p, model, 0)


def test_original_prefix_bytes_survive_later_continuation(tmp_path):
    old = tmp_path / 'old.jsonl'; recovery = tmp_path / 'recovery.jsonl'
    old.write_bytes(b'{"answer": "2"}\n')
    recovery.write_bytes(old.read_bytes() + b'{"answer": "3"}\n')
    resume.require_original_prefix(recovery, old)
    recovery.write_bytes(b'{"answer": "3"}\n' + b'{"answer": "3"}\n')
    with pytest.raises(AssertionError, match='bytes changed'):
        resume.require_original_prefix(recovery, old)


def test_replay_permits_only_clock_changes(sample):
    _, _, _, _, row = sample
    replay = copy.deepcopy(row)
    replay['observation']['started_at'] = 'now'
    replay['observation']['finished_at'] = 'later'
    replay['observation']['answers'][0]['latency_ms'] = 999
    resume.require_replay_match(row, replay)
    # Same normalized value is insufficient: original raw output must repeat.
    replay['observation']['answers'][0]['raw_answer'] = 'two'
    with pytest.raises(AssertionError, match='replay differs'):
        resume.require_replay_match(row, replay)


def test_completed_skip_requires_all_rows_hash_count_and_parameter_digest(tmp_path, sample):
    scene, image, p, model, row = sample
    rows = []
    for cond in p['conditions']:
        r = copy.deepcopy(row); r['condition'] = cond; rows.append(r)
    path = tmp_path / 'answers.jsonl'; write_rows(path, rows)
    marker = {'n_images': 1, 'n_answers': 3, 'answers_sha256': resume.sha(path),
        'parameter_digest_before': 'parameters', 'parameter_digest_after': 'parameters'}
    (tmp_path / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
    expected = resume.expected_sequence([image], p['conditions'])
    resume.validate_complete(tmp_path, rows, 3, expected, model, 1)
    with pytest.raises(AssertionError, match='incomplete rows'):
        resume.validate_complete(tmp_path, rows[:-1], 2, expected, model, 1)
    marker['parameter_digest_after'] = 'changed'
    (tmp_path / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
    with pytest.raises(AssertionError):
        resume.validate_complete(tmp_path, rows, 3, expected, model, 1)


def test_questions_keep_original_condition_semantics(sample):
    scene, image, _, _, _ = sample
    off = resume.question_set(scene, image, 'prompt_off')[0]
    blank = resume.question_set(scene, image, 'prompt_blank')[0]
    on = resume.question_set(scene, image, 'prompt_on')[0]
    assert off.text == scene['questions'][0]['text']
    assert 'two red books' not in blank.text and 'two red books' in on.text
    assert off.question_id == blank.question_id == on.question_id
    assert off.expected_answer == blank.expected_answer == on.expected_answer == '2'

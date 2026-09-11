"""Resume the frozen Stage B observation sequence without replacing old answers."""
from runtime_common import *
import argparse
import os
from dataclasses import replace


def question_set(scene, row, condition):
    from selfsight.schemas import AtomicQuestion
    from selfsight.v4.observe import PROMPTED_PREAMBLE
    raw = tuple(AtomicQuestion.from_dict(q) for q in scene['questions'])
    if condition == 'prompt_off':
        return raw
    assert condition in ('prompt_on', 'prompt_blank')
    return tuple(replace(q, text=PROMPTED_PREAMBLE.format(
        prompt=row['prompt'] if condition == 'prompt_on' else '', question=q.text)) for q in raw)


def expected_sequence(images, conditions):
    return [(row, condition) for row in images for condition in conditions]


def validate_row(saved, row, condition, scene, p, model, shard):
    from selfsight.data.questions import normalize_answer
    identity = {'observer': model['id'], 'generator': 'base', 'spec_id': row['spec_id'],
        'scene_sha256': row['scene_sha256'], 'candidate_index': row['candidate_index'],
        'seed': row['seed'], 'condition': condition, 'n_original': scene['n_original'],
        'image_path': row['image_path'], 'device': f'cuda:{shard}'}
    for key, value in identity.items():
        assert saved[key] == value, f'Prefix identity mismatch: {key}'
    assert row['scene_sha256'] == scene['scene_sha256']
    assert row['prompt'] == scene['spec']['prompt']
    obs = saved['observation']
    assert obs['rgb_sha256'] == row['rgb_sha256'], 'RGB mismatch'
    assert obs['observer_id'] == 'showlab/show-o2-1.5B'
    assert obs['observer_revision'] == p['observer_revision']
    qs = question_set(scene, row, condition)
    assert len(obs['answers']) == len(qs), 'Partial condition is not resumable'
    for q, a in zip(qs, obs['answers']):
        assert a['question_id'] == q.question_id, 'Question order mismatch'
        assert not a.get('error'), 'Observer error remains a hard stop'
        normalized = normalize_answer(a['raw_answer'], q)
        assert normalized == a['normalized_answer'], 'Normalization mismatch'
        assert a['abstain'] is (normalized is None), 'Abstention mismatch'
    return len(qs)


def validate_prefix(path, expected, scenes, p, model, shard):
    path = Path(path)
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    assert not data or data.endswith(b'\n'), 'Incomplete JSONL suffix; do not discard it'
    raw_lines = data.splitlines()
    assert all(line.strip() for line in raw_lines), 'Blank prefix records are invalid'
    rows = [json.loads(line) for line in raw_lines]
    assert len(rows) <= len(expected), 'Prefix exceeds frozen sequence'
    n = 0
    for saved, (row, condition) in zip(rows, expected):
        n += validate_row(saved, row, condition, scenes[row['spec_id']], p, model, shard)
    return rows, n


def require_original_prefix(path, original_path):
    if Path(original_path).exists():
        frozen = Path(original_path).read_bytes()
        assert Path(path).exists(), 'Original observations were not copied'
        assert Path(path).read_bytes()[:len(frozen)] == frozen, 'Original observation bytes changed'


def validate_complete(folder, rows, n, expected, model, n_images):
    done = read(folder / 'complete.json')
    assert len(rows) == len(expected), 'Complete marker has incomplete rows'
    assert done['n_images'] == n_images and done['n_answers'] == n
    assert done['answers_sha256'] == sha(folder / 'answers.jsonl'), 'Complete hash mismatch'
    assert done['parameter_digest_before'] == done['parameter_digest_after'] == model['parameter_digest']


def semantic_observation(obs):
    # Wall-clock timing/latency is not repeatable and never enters the endpoint.
    return {key: obs[key] for key in ('request_id', 'observer_id', 'observer_revision', 'rgb_sha256')} | {
        'answers': [{key: a.get(key) for key in
            ('question_id', 'raw_answer', 'normalized_answer', 'abstain', 'error')} for a in obs['answers']]}


def require_replay_match(previous, replay):
    assert semantic_observation(previous['observation']) == semantic_observation(replay['observation']), \
        'Resume boundary replay differs; stop without appending'


def observe_one(backbone, model, row, condition, scene, shard):
    from selfsight.schemas import as_serializable
    from selfsight.training.checkpoint import _rng_state, _restore_rng_state
    from selfsight.v4.train import seed_training
    rng = _rng_state()
    seed_training(seed('B-observe', row['spec_id'], row['candidate_index']))
    try:
        observation = backbone.observe_atoms(row['image_path'], question_set(scene, row, condition))
    finally:
        _restore_rng_state(rng)
    return {'observer': model['id'], 'generator': 'base', 'spec_id': row['spec_id'],
        'scene_sha256': row['scene_sha256'], 'candidate_index': row['candidate_index'],
        'seed': row['seed'], 'condition': condition, 'n_original': scene['n_original'],
        'image_path': row['image_path'], 'device': f'cuda:{shard}', 'observation': as_serializable(observation)}


def prepare_inputs(p, shard):
    from selfsight.utils.hashing import rgb_sha256
    scenes = {r['spec']['spec_id']: r for r in scene_rows()}
    assigned = set(p['observation_shards'][str(shard)])
    images = [r for r in all_images(p) if r['spec_id'] in assigned]
    assert len(images) == 6 * len(assigned)
    assert len({(r['spec_id'], r['candidate_index']) for r in images}) == len(images)
    for row in images:
        scene = scenes[row['spec_id']]
        assert row['spec'] == scene['spec'] and row['scene_sha256'] == scene['scene_sha256']
        assert row['seed'] == scene['sampling_seeds'][row['candidate_index']]
        assert sha(row['image_path']) == row['file_sha256']
        assert rgb_sha256(row['image_path']) == row['rgb_sha256']
    expected = expected_sequence(images, p['conditions'])
    states = []
    for model in p['models']:
        folder = R / 'observations' / f'gpu{shard}' / model['id']
        require_original_prefix(folder / 'answers.jsonl',
            OLD / 'observations' / f'gpu{shard}' / model['id'] / 'answers.jsonl')
        rows, n = validate_prefix(folder / 'answers.jsonl', expected, scenes, p, model, shard)
        complete = (folder / 'complete.json').exists()
        if complete:
            validate_complete(folder, rows, n, expected, model, len(images))
        states.append((model, folder, rows, n, complete))
    return scenes, assigned, images, expected, states


def observe(p, deadline, shard, validate_only=False):
    scenes, assigned, images, expected, states = prepare_inputs(p, shard)
    print(json.dumps({'shard': shard, 'validated': [
        {'model': m['id'], 'prefix_conditions': len(rs), 'prefix_answers': n, 'complete': done}
        for m, _, rs, n, done in states]}), flush=True)
    if validate_only:
        return
    check_time(deadline)
    if all(done for _, _, _, _, done in states):
        marker = R / 'observations' / f'gpu{shard}' / 'complete.json'
        if not marker.exists():
            save_new(marker, {'at_utc': now(), 'n_scenes': len(assigned), 'models': len(p['models'])})
        return
    device = f'cuda:{shard}'
    wait_room(device, deadline)
    from selfsight.v4.train import parameter_digest, trainable_snapshot
    backbone, config = make_backbone(p, device)
    for model, folder, prefix, n, complete in states:
        if complete:
            print(f'{device} {model["id"]}: frozen complete output verified; skipped', flush=True)
            continue
        check_time(deadline)
        before = load_model(backbone, model, config)
        assert before == model['parameter_digest']
        folder.mkdir(parents=True, exist_ok=True)
        token = f'{time.time_ns()}-{os.getpid()}'
        answer_path = folder / 'answers.jsonl'
        prefix_bytes = answer_path.read_bytes() if answer_path.exists() else b''
        save_new(folder / f'resume-entry-{token}.json', {'at_utc': now(), 'prefix_conditions': len(prefix),
            'prefix_answers': n, 'prefix_bytes': len(prefix_bytes),
            'prefix_sha256': hashlib.sha256(prefix_bytes).hexdigest(), 'parameter_digest': before,
            'deadline': deadline, 'original_plan_sha256': sha(OLD / 'plan.json')})
        if prefix:
            check_time(deadline)
            row, condition = expected[len(prefix) - 1]
            scene = scenes[row['spec_id']]
            replay = observe_one(backbone, model, row, condition, scene, shard)
            # Save even a mismatch for diagnosis; replay never enters answers.jsonl.
            save_new(folder / f'replay-{token}.json', {'at_utc': now(), 'excluded_from_main_denominator': True,
                'prefix_condition_index': len(prefix) - 1, 'previous': prefix[-1], 'replay': replay})
            validate_row(replay, row, condition, scene, p, model, shard)
            require_replay_match(prefix[-1], replay)
            save_new(folder / f'replay-pass-{token}.json', {'at_utc': now(), 'all_semantic_fields_equal': True,
                'n_answers': len(replay['observation']['answers'])})
            print(f'{device} {model["id"]}: boundary replay exact; resuming at condition {len(prefix)}', flush=True)
        # Refuse a simultaneous writer rather than append to an unverified tail.
        assert (answer_path.read_bytes() if answer_path.exists() else b'') == prefix_bytes
        with answer_path.open('a' if answer_path.exists() else 'x', encoding='utf-8') as log:
            for index in range(len(prefix), len(expected)):
                check_time(deadline)
                row, condition = expected[index]
                scene = scenes[row['spec_id']]
                saved = observe_one(backbone, model, row, condition, scene, shard)
                append(log, saved)
                n += validate_row(saved, row, condition, scene, p, model, shard)
                if (index + 1) % 36 == 0:
                    print(f'{device} {model["id"]}: {(index + 1) // 3}/{len(images)} images, {n} answers', flush=True)
        after = parameter_digest(trainable_snapshot(backbone.model))
        assert after == before
        assert answer_path.read_bytes()[:len(prefix_bytes)] == prefix_bytes
        rows, checked_n = validate_prefix(answer_path, expected, scenes, p, model, shard)
        assert len(rows) == len(expected) and checked_n == n
        save_new(folder / 'complete.json', {'at_utc': now(), 'n_images': len(images), 'n_answers': n,
            'parameter_digest_before': before, 'parameter_digest_after': after, 'answers_sha256': sha(answer_path)})
    save_new(R / 'observations' / f'gpu{shard}' / 'complete.json',
        {'at_utc': now(), 'n_scenes': len(assigned), 'models': len(p['models'])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, choices=(0, 1), required=True)
    ap.add_argument('--deadline', type=float, required=True)
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    verify_recovery()
    p = original_plan()
    observe(p, a.deadline, a.shard, a.validate_only)


if __name__ == '__main__':
    main()

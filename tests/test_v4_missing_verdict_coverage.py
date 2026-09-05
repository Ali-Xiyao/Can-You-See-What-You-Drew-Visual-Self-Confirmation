"""A detector error cannot silently shrink the generated-image denominator."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from selfsight.schemas import CandidateRecord
from selfsight.v4.evaluate import external_correctness, read_metrics_csv
from selfsight.v4.train import read_verdicts, select_gold


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def fixture_96(tmp_path):
    images = [str(tmp_path / f'image-{i:02}.png') for i in range(96)]
    missing = 46
    pending = {1, 9, 17, 25, 33, 41}
    manifest = [{'image_path': image, 'spec_id': f'p{i // 8}', 'candidate_index': i % 8}
                for i, image in enumerate(images)]
    primary = [{'image_path': image, **({'error': 'json_error: truncated output'} if i == missing
                else {'detections': []})} for i, image in enumerate(images)]
    # This is exactly stage_verify's coverage behavior: the error row has no
    # detections, so cached_detections omits it and no verdict gets written.
    success = {row['image_path'] for row in primary if 'detections' in row}
    verified = [{'image_path': image, 'image_correct': i % 8 == 0,
                 'resolution': 'pending_human' if i in pending else 'agreed'}
                for i, image in enumerate(images) if image in success]
    write_jsonl(tmp_path / 'manifest.jsonl', manifest)
    write_jsonl(tmp_path / 'detections.qwen3vl.jsonl', primary)
    write_jsonl(tmp_path / 'verified.jsonl', verified)
    pools = {}
    for i, image in enumerate(images):
        prompt = f'p{i // 8}'
        pools.setdefault(prompt, []).append(CandidateRecord(
            candidate_id=f'{prompt}:{i % 8}', prompt_id=prompt, scene_id=prompt,
            sampling_seed=i, image_path=image, rgb_sha256='hash', generator_id='fixture',
            generator_revision='fixture', checkpoint_id='fixture'))
    return images, missing, pending, pools


def test_96_to_95_detector_error_is_unknown_and_valid_gold_choices_stay_identical(tmp_path):
    images, missing, pending, pools = fixture_96(tmp_path)
    verdicts, unknown = read_verdicts(tmp_path / 'verified.jsonl', pools)
    assert len(unknown) == 7
    assert unknown == {images[i] for i in pending | {missing}}
    assert 'p5:6' not in verdicts, 'missing is not a negative verdict'
    chosen = select_gold(pools=pools, verdicts=verdicts)
    assert {d.selected_candidate_id for d in chosen} == {f'p{i}:0' for i in range(12)}
    assert all(verdicts[d.selected_candidate_id] is True for d in chosen)
    legacy_known_verdicts = {f'p{i // 8}:{i % 8}': i % 8 == 0
                             for i in range(96) if i != missing and i not in pending}
    assert chosen == select_gold(pools=pools, verdicts=legacy_known_verdicts)


def test_manifest_coverage_preserves_adjudicated_rate_without_counting_missing_wrong(tmp_path):
    fixture_96(tmp_path)
    path = tmp_path / 'verified.jsonl'
    rate, n, unknown = external_correctness(path, manifest_path=tmp_path / 'manifest.jsonl')
    assert n == 89 and unknown == 7 and n + unknown == 96
    assert rate == pytest.approx(12 / 89)
    # Historical one-argument callers still obtain the historical 95-row coverage.
    assert external_correctness(path) == (rate, 89, 6)


def test_completely_missing_verified_file_is_all_unknown_when_manifest_is_known(tmp_path):
    fixture_96(tmp_path)
    path = tmp_path / 'never-produced.jsonl'
    assert external_correctness(path, manifest_path=tmp_path / 'manifest.jsonl') == (None, 0, 96)
    assert external_correctness(path) == (None, 0, 0)


def test_stage_score_csv_keeps_full_manifest_coverage_after_detector_error(tmp_path):
    directory = tmp_path / 'evaluations/naive/step-00008'
    directory.mkdir(parents=True)
    fixture_96(directory)
    (directory / 'cycle.json').write_text(json.dumps(
        {'arm': 'naive', 'round': 0, 'step': 8, 'mean': -1.0, 'sem': 0.01, 'n': 96}), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts/v4_train.py'
    spec = importlib.util.spec_from_file_location('coverage_v4_train_script', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = {name: (directory / name).read_bytes() for name in
              ('manifest.jsonl', 'detections.qwen3vl.jsonl', 'verified.jsonl')}
    module.stage_score(argparse.Namespace(outdir=tmp_path))
    row, = read_metrics_csv(tmp_path / 'checkpoint_metrics.csv')
    assert row.external_n == 89 and row.external_unadjudicated == 7
    assert row.external_n + row.external_unadjudicated == 96
    assert row.external_coverage_policy == 'manifest_image_verdict_v1'
    assert row.external_correct == pytest.approx(12 / 89)
    assert before == {name: (directory / name).read_bytes() for name in before}
    # The next invocation reads its own CSV, including the string policy,
    # before replacing the same checkpoint row rather than duplicating it.
    module.stage_score(argparse.Namespace(outdir=tmp_path))
    assert read_metrics_csv(tmp_path / 'checkpoint_metrics.csv') == [row]


def test_legacy_csv_without_policy_column_remains_readable(tmp_path):
    path = tmp_path / 'legacy.csv'
    fields = ['arm', 'round_index', 'step', 'internal_cycle', 'internal_sem', 'internal_n',
              'external_correct', 'external_n', 'external_unadjudicated']
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(arm='naive', round_index=0, step=8, internal_cycle=-1,
                             internal_sem=0.01, internal_n=96, external_correct=0.5,
                             external_n=88, external_unadjudicated=7))
    row, = read_metrics_csv(path)
    assert row.external_coverage_policy == 'legacy_row_only'
    assert row.external_correct == 0.5 and row.external_n == 88


def test_unnameable_false_is_known_failure_but_nonboolean_and_pending_are_unknown(tmp_path):
    images, missing, pending, pools = fixture_96(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / 'verified.jsonl').read_text().splitlines()]
    # Index2 was already known False. Its facts are unnameable, but changing
    # the resolution must not improve the image-level generation success rate.
    rows[2]['resolution'] = 'unnameable'
    assert rows[2]['image_correct'] is False
    rows[3]['image_correct'] = 'false'
    rows[4]['image_correct'] = None
    write_jsonl(tmp_path / 'verified.jsonl', rows)
    verdicts, unknown = read_verdicts(tmp_path / 'verified.jsonl', pools)
    assert verdicts['p0:2'] is False and images[2] not in unknown
    assert images[3] in unknown and images[4] in unknown
    rate, n, unavailable = external_correctness(tmp_path / 'verified.jsonl',
                                               manifest_path=tmp_path / 'manifest.jsonl')
    assert (n, unavailable) == (87, 9)
    assert rate == pytest.approx(12 / 87)


def test_unnameable_false_csv_counts_known_failure_and_legacy_is_explicit(tmp_path):
    directory = tmp_path / 'evaluations/naive/step-00000'
    directory.mkdir(parents=True)
    write_jsonl(directory / 'manifest.jsonl', [{'image_path': 'a.png'}, {'image_path': 'b.png'}])
    write_jsonl(directory / 'verified.jsonl', [
        {'image_path': 'a.png', 'resolution': 'agreed', 'image_correct': True},
        {'image_path': 'b.png', 'resolution': 'unnameable', 'image_correct': False}])
    (directory / 'cycle.json').write_text(json.dumps(
        {'arm': 'naive', 'round': -1, 'step': 0, 'mean': -1.0, 'sem': 0.01, 'n': 2}), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts/v4_train.py'
    spec = importlib.util.spec_from_file_location('coverage_unnameable_script', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.stage_score(argparse.Namespace(outdir=tmp_path))
    row, = read_metrics_csv(tmp_path / 'checkpoint_metrics.csv')
    assert (row.external_correct, row.external_n, row.external_unadjudicated) == (0.5, 2, 0)
    assert row.external_coverage_policy == 'manifest_image_verdict_v1'
    # The old function call is preserved, but the new pipeline never uses it
    # when its generated-image manifest exists.
    assert external_correctness(directory / 'verified.jsonl') == (1.0, 1, 1)


@pytest.mark.parametrize('malformation', ['duplicate_manifest', 'duplicate_verdict', 'foreign_verdict'])
def test_manifest_mode_refuses_ambiguous_image_universe(tmp_path, malformation):
    fixture_96(tmp_path)
    target = tmp_path / ('manifest.jsonl' if malformation == 'duplicate_manifest' else 'verified.jsonl')
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    rows.append(dict(rows[0], image_path='other.png') if malformation == 'foreign_verdict' else rows[0])
    write_jsonl(target, rows)
    with pytest.raises(ValueError):
        external_correctness(tmp_path / 'verified.jsonl', manifest_path=tmp_path / 'manifest.jsonl')

"""CPU regression checks for recovery's lossless logging and immediate stop."""
import io
import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import detector_recovery as recovery
from PIL import Image
from selfsight.v4.detectors import INSTRUCTION, VlmDetector


class DetectorRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent
        self.folder = self.test_root / ('test-detector-' + uuid.uuid4().hex)
        self.folder.mkdir()
        self.image = self.folder / 'test.png'
        Image.new('RGB', (12, 18), (42, 16, 81)).save(self.image)
        self.time_patch = patch.object(recovery, 'check_time')
        self.time_patch.start()

    def tearDown(self):
        self.time_patch.stop()
        assert self.folder.resolve().parent == self.test_root
        shutil.rmtree(self.folder)

    def test_success_preserves_full_raw_and_frozen_input(self):
        raw = '[{"object":"book","color":"red","box":[0,0,1,1]}]'
        log = io.StringIO()
        original = VlmDetector('fake', lambda image, instruction: raw)
        wrapped = recovery.AuditedDetector(original, log, 99, 'fake', 'detect')
        result = wrapped.detect(str(self.image))
        event = json.loads(log.getvalue())
        self.assertEqual(result[0]['object'], 'book')
        self.assertEqual(event['reply'], raw)
        self.assertEqual(event['instruction'], INSTRUCTION)
        self.assertEqual(event['input']['width'], 12)
        self.assertEqual(event['input']['height'], 18)
        self.assertEqual(event['status'], 'ok')

    def test_parse_failure_saves_untruncated_reply_and_reaches_original_handler(self):
        raw = 'x' * 900 + '[{"object":"book"}, broken]'
        log = io.StringIO()
        wrapped = recovery.AuditedDetector(VlmDetector('fake', lambda *_: raw), log, 99, 'fake', 'detect')
        with self.assertRaises(ValueError):
            wrapped.detect(str(self.image))
        events = [json.loads(x) for x in log.getvalue().splitlines()]
        self.assertEqual(events[0]['reply'], raw)
        self.assertEqual(events[0]['status'], 'parse_error')
        self.assertTrue(events[0]['error'].startswith('json_error:'))
        self.assertEqual(events[1]['event'], 'operation_error')

    def test_json_failure_is_quarantined_and_next_image_runs_without_retry(self):
        second = self.folder / 'second.png'
        second.write_bytes(self.image.read_bytes())
        manifest = self.folder / 'manifest.jsonl'
        data = [{'image_path': str(p)} for p in [self.image, second]]
        manifest.write_text(''.join(json.dumps(r) + '\n' for r in data))
        replies = iter(['[{"object":"book"}, broken]', '[]'])
        raw = '[{"object":"book"}, broken]'
        calls = []
        def run(*_):
            calls.append(1)
            return next(replies)
        pipeline = recovery.pipeline_module()
        pipeline.await_room = lambda *_: None
        out = self.folder / 'detections.qwen3vl.jsonl'
        with patch('selfsight.v4.detectors.load', return_value=VlmDetector('fake', run)):
            with recovery.audited_loader(self.folder, 'detect', 99):
                pipeline.stage_detect(SimpleNamespace(manifest=manifest, output=out,
                    overwrite=False, detector='qwen3vl', device='cpu'))
        recovery.annotate_quarantines(out, self.folder / 'raw-calls.detect.qwen3vl.jsonl')
        rows = recovery.validate_detections(out, data)
        self.assertEqual(len(calls), 2)
        self.assertTrue(rows[0]['quarantined'])
        self.assertEqual(rows[0]['reply'], raw)
        self.assertNotIn('detections', rows[0])
        self.assertEqual(rows[0]['raw_capture']['call_index'], 1)
        self.assertEqual(rows[1]['detections'], [])
        self.assertTrue(out.with_name(out.name + '.pipeline-original').exists())

    def test_missing_detector_never_becomes_single_detector_or_empty_scene_gold(self):
        from selfsight.v4.factual_truth import factual_answer
        data = [{'image_path': 'a', 'spec_id': 'scene', 'candidate_index': 0, 'seed': 10,
                 'spec': {'spec_id': 'scene', 'prompt': 'a red book',
                          'objects': [{'object': 'book', 'color': 'red', 'count': 1}]}}]
        good = [{'image_path': 'a', 'detections': [{'object': 'book', 'color': 'red'}]}]
        failed = [{'image_path': 'a', 'error': 'json_error', 'quarantined': True,
                   'failure_class': 'json_parse_error', 'reply': '[{"object":"book"}, broken]'}]
        pipeline = recovery.pipeline_module()
        rows = recovery.verify_rows(data, good, failed, {}, pipeline)
        row = rows[0]
        self.assertIsNone(row['image_correct'])
        self.assertEqual(row['resolution'], 'pending_human')
        self.assertTrue(row['factual_truth_unavailable'])
        self.assertEqual(row['missing_detectors'], ['internvl'])
        self.assertFalse(factual_answer({'family': 'count', 'text': 'How many red books are in this picture?'}, row).known)
        complete = recovery.verify_rows(data, good, good, {}, pipeline)[0]
        self.assertTrue(complete['image_correct'])
        self.assertEqual(complete['resolution'], 'agreed')

    def test_quarantine_cannot_hide_a_non_json_failure_or_supply_detections(self):
        for reply in ['no array', '[]']:
            with self.subTest(reply=reply), self.assertRaises(AssertionError):
                recovery.validate_row({'error': 'bad', 'quarantined': True,
                    'failure_class': 'json_parse_error', 'reply': reply})
        with self.assertRaises(AssertionError):
            recovery.validate_row({'error': 'bad', 'quarantined': True,
                'failure_class': 'json_parse_error', 'reply': '[bad]', 'detections': []})

    def test_model_exception_never_reuses_previous_reply(self):
        replies = iter(['[]', RuntimeError('model failed')])
        def run(*_):
            item = next(replies)
            if isinstance(item, Exception):
                raise item
            return item
        log = io.StringIO()
        wrapped = recovery.AuditedDetector(VlmDetector('fake', run), log, 99, 'fake', 'detect')
        wrapped.detect(str(self.image))
        with self.assertRaises(recovery.DetectorStopped):
            wrapped.detect(str(self.image))
        events = [json.loads(x) for x in log.getvalue().splitlines()]
        self.assertIsNone(events[1]['reply'])
        self.assertEqual(events[1]['status'], 'model_error')
        self.assertEqual(events[2]['last_reply'], '')

    def test_original_detect_pipeline_cannot_continue_after_parse_error(self):
        manifest = self.folder / 'manifest.jsonl'
        manifest.write_text(json.dumps({'image_path': str(self.image)}) + '\n' +
                            json.dumps({'image_path': 'never-visited.png'}) + '\n')
        pipeline = recovery.pipeline_module()
        pipeline.await_room = lambda *_: None
        calls = []
        def run(*_):
            calls.append(1)
            return 'invalid'
        with patch('selfsight.v4.detectors.load', return_value=VlmDetector('fake', run)):
            with recovery.audited_loader(self.folder, 'detect', 99):
                with self.assertRaises(recovery.DetectorStopped):
                    pipeline.stage_detect(SimpleNamespace(manifest=manifest,
                        output=self.folder / 'out.jsonl', overwrite=False,
                        detector='qwen3vl', device='cpu'))
        self.assertEqual(len(calls), 1)
        events = recovery.lines(self.folder / 'raw-calls.detect.qwen3vl.jsonl')
        self.assertEqual(events[0]['reply'], 'invalid')

    def test_crop_exception_is_not_swallowed_by_legacy_handlers(self):
        log = io.StringIO()
        wrapped = recovery.AuditedDetector(VlmDetector('fake', lambda *_: 'bad'), log, 99, 'fake', 'crop')
        with self.assertRaises(recovery.DetectorStopped):
            wrapped.detect_crop(str(self.image), (1, 2, 10, 16))
        event = json.loads(log.getvalue().splitlines()[0])
        self.assertEqual(event['bbox'], [1, 2, 10, 16])
        self.assertEqual(event['input']['width'], 36)
        self.assertEqual(event['input']['height'], 56)
        self.assertNotIsInstance(recovery.DetectorStopped(), (ValueError, OSError, RuntimeError))

    def test_cached_outputs_require_exact_unique_complete_clean_ids(self):
        path = self.folder / 'out.jsonl'
        data = [{'image_path': 'a'}, {'image_path': 'b'}]
        cases = [
            [{'image_path': 'a', 'detections': []}],
            [{'image_path': 'a', 'detections': []}] * 2,
            [{'image_path': 'a', 'detections': []}, {'image_path': 'c', 'detections': []}],
            [{'image_path': 'a', 'detections': []}, {'image_path': 'b', 'error': 'bad'}],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.assertRaises(AssertionError):
                    recovery.validate_detections(path, data)
        valid = [{'image_path': 'b', 'detections': []}, {'image_path': 'a', 'detections': []}]
        path.write_text(''.join(json.dumps(r) + '\n' for r in valid))
        self.assertEqual(recovery.validate_detections(path, data), valid)


if __name__ == '__main__':
    unittest.main()

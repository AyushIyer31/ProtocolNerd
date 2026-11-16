"""Ignore non-object JSON lines without losing usable local or mirrored records."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'protocolnerd-backend' / 'query_logger.py'
spec = importlib.util.spec_from_file_location('query_log_shapes', PATH)
ql = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ql)

class RecordShapeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.date = '2026-11-15'
        self.local = self.root / ('query_results_' + self.date + '.log')
        for name, value in {'LOG_DIR': self.root, 'S3_BUCKET': '',
                            '_s3_broken': False, '_s3_client': None}.items():
            patcher = patch.object(ql, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def read(self, lines):
        self.local.write_text('\n'.join(lines), encoding='utf-8')
        return ql.read_records(self.date)

    def check_scalar(self, value):
        good = {'ts': '2026-11-15T12:00:00Z', 'event': 'search', 'session_id': 's'}
        self.assertEqual(self.read([json.dumps(value), json.dumps(good)]), [good])

    def test_skip_null(self): self.check_scalar(None)
    def test_skip_true(self): self.check_scalar(True)
    def test_skip_false(self): self.check_scalar(False)
    def test_skip_integer(self): self.check_scalar(4)
    def test_skip_float(self): self.check_scalar(1.25)
    def test_skip_string(self): self.check_scalar('hello')
    def test_skip_empty_list(self): self.check_scalar([])
    def test_skip_object_list(self): self.check_scalar([{'ts': 'bad'}])
    def test_skip_nested_list(self): self.check_scalar([[1]])

    def test_malformed_and_blank_lines(self):
        self.assertEqual(self.read(['{', '', ' ', '{"ts":"a"}']), [{'ts': 'a'}])

    def test_objects_keep_fields_and_order(self):
        records = [{'ts': 'b', 'event': 'search', 'extra': {'a': [1]}}, {'ts': 'a'}]
        self.assertEqual(self.read(list(map(json.dumps, records))), records[::-1])

    def test_duplicate_objects_still_collapse(self):
        self.assertEqual(self.read(['{"ts":"a"}', '{"ts":"a"}']), [{'ts': 'a'}])

    def test_object_without_timestamp_still_accepted(self):
        self.assertEqual(self.read(['{"event":"new_query"}']), [{'event': 'new_query'}])

    def test_all_nonobjects_returns_empty(self):
        self.assertEqual(self.read(['null', '[]', 'false', '1', '"text"']), [])

    def test_mirrored_nonobjects_do_not_hide_local_records(self):
        date = self.date
        class S3:
            def get_paginator(self, name):
                class Pages:
                    def paginate(self, **kwargs):
                        yield {'Contents': [{'Key': 'query_logs/query_results_' + date + '.writer.log'}]}
                return Pages()
            def get_object(self, **kwargs):
                return {'Body': io.BytesIO(b'null\n[]\n{"ts":"a","event":"search"}\n')}
        with patch.object(ql, 'S3_BUCKET', 'fake'), patch.object(ql, '_s3_client', S3()):
            self.assertEqual(self.read(['{"ts":"b"}']), [{'ts': 'a', 'event': 'search'}, {'ts': 'b'}])

if __name__ == '__main__':
    unittest.main()

"""The S3 mirror of the query log: records reach S3 without another request, tasks that
overlap do not overwrite each other, and the viewer merges every source."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp()
os.environ.update({
    "QUERY_LOG_DIR": _TMP,
    "QUERY_LOG_S3_BUCKET": "test-bucket",
    "QUERY_LOG_S3_PREFIX": "query_logs/",
    "QUERY_LOG_S3_WRITER_ID": "writerA",
    "QUERY_LOG_S3_FLUSH_SECONDS": "3600",        # the tests flush by hand
})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "protocolnerd-backend"))
import query_logger as ql  # noqa: E402


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.fail_puts = 0

    def put_object(self, Bucket, Key, Body, ContentType=None):
        if self.fail_puts:
            self.fail_puts -= 1
            raise RuntimeError("simulated S3 outage")
        self.objects[Key] = bytes(Body)

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        objs = self.objects

        class P:
            def paginate(self, Bucket, Prefix=""):
                yield {"Contents": [{"Key": k} for k in sorted(objs) if k.startswith(Prefix)]}
        return P()


def _line(ts, session, query):
    return json.dumps({"ts": ts, "event": "search", "session_id": session, "original_query": query})


class QueryLoggerS3Test(unittest.TestCase):
    def setUp(self):
        self.s3 = FakeS3()
        ql._s3_client = self.s3
        ql._s3_broken = False
        with ql._lock:
            ql._dirty.clear()
        for f in Path(_TMP).glob("*.log"):
            f.unlink()
        self.date = ql._today()
        self.local = Path(_TMP) / f"query_results_{self.date}.log"

    def test_record_reaches_s3_on_flush_without_another_request(self):
        ql.log_event("search", session_id="s1", original_query="drought rice")
        self.assertEqual(self.local.read_text().count("\n"), 1)
        self.assertEqual(self.s3.objects, {}, "the request path itself must not upload")
        ql._flush_dirty()
        key = f"query_logs/query_results_{self.date}.writerA.log"
        self.assertIn(key, self.s3.objects)
        self.assertIn("drought rice", self.s3.objects[key].decode())
        self.assertEqual(ql._dirty, set())

    def test_overlapping_writers_and_local_records_all_appear_once_in_time_order(self):
        # another task's object, a legacy unsuffixed object, and this task's own upload
        self.s3.objects[f"query_logs/query_results_{self.date}.writerB.log"] = (
            _line("2026-09-24T10:00:00+00:00", "b", "from writer B") + "\n").encode()
        self.s3.objects[f"query_logs/query_results_{self.date}.log"] = (
            _line("2026-09-24T09:00:00+00:00", "l", "legacy object") + "\n").encode()
        ql.log_event("search", session_id="a1", original_query="uploaded by A")
        ql._flush_dirty()
        ql.log_event("search", session_id="a2", original_query="still local on A")
        recs = ql.read_records(self.date)
        queries = [r["original_query"] for r in recs]
        self.assertEqual(len(recs), 4, queries)
        self.assertEqual(queries[:2], ["legacy object", "from writer B"])
        self.assertEqual(queries[2:], ["uploaded by A", "still local on A"])
        self.assertEqual(queries.count("uploaded by A"), 1, "uploaded and on disk: once")

    def test_list_dates_sees_suffixed_and_legacy_objects(self):
        self.s3.objects["query_logs/query_results_2026-01-02.abc123.log"] = b""
        self.s3.objects["query_logs/query_results_2026-01-01.log"] = b""
        self.s3.objects["query_logs/notes.txt"] = b""
        self.assertEqual(ql.list_dates(), ["2026-01-02", "2026-01-01"])

    def test_failed_upload_stays_dirty_and_retries(self):
        ql.log_event("search", session_id="s1", original_query="q")
        self.s3.fail_puts = 1
        ql._flush_dirty()
        self.assertEqual(ql._dirty, {self.local})
        ql._flush_dirty()
        self.assertEqual(ql._dirty, set())
        self.assertEqual(len(self.s3.objects), 1)

    def test_rejects_non_date(self):
        self.assertEqual(ql.read_records("../etc/passwd"), [])


if __name__ == "__main__":
    unittest.main()

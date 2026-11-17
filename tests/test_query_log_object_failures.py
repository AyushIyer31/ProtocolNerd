"""A failed mirrored writer must not hide records from other writers. Offline only."""
import importlib.util
import io
import json
from pathlib import Path
import socket

import pytest

PATH = Path(__file__).resolve().parents[1] / 'protocolnerd-backend' / 'query_logger.py'

@pytest.fixture
def logs(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError('Network is not permitted in query-log tests')
    monkeypatch.setattr(socket, 'create_connection', no_network)
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    spec = importlib.util.spec_from_file_location('query_log_object_failures', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'LOG_DIR', tmp_path)
    monkeypatch.setattr(module, 'S3_BUCKET', 'synthetic-bucket')
    monkeypatch.setattr(module, 'S3_PREFIX', 'query_logs/')
    monkeypatch.setattr(module, '_s3_broken', False)
    return module

DATE = '2026-11-17'

def encoded(ts, source):
    return (json.dumps({'ts': ts, 'source': source}) + '\n').encode()

class FakeS3:
    def __init__(self, items):
        self.items = items
        self.read_keys = []
    def get_paginator(self, name):
        items = self.items
        class Pages:
            def paginate(self, **kwargs):
                # Separate pages make it observable if an earlier failed object
                # prevents a later page from being visited.
                for key, body in items:
                    yield {'Contents': [{'Key': key}]}
        return Pages()
    def get_object(self, **kwargs):
        key = kwargs['Key']
        self.read_keys.append(key)
        body = dict(self.items)[key]
        if isinstance(body, Exception):
            raise body
        if body == 'read-error':
            class BrokenBody:
                def read(self):
                    raise OSError('synthetic body read failure')
            return {'Body': BrokenBody()}
        return {'Body': io.BytesIO(body)}

def key(name):
    return 'query_logs/query_results_' + DATE + '.' + name + '.log'

@pytest.mark.parametrize('position', [0, 1, 2], ids=['first', 'middle', 'last'])
@pytest.mark.parametrize('failure', [RuntimeError('synthetic missing object'),
                                    PermissionError('synthetic denied object'),
                                    'read-error', b'\xff\xfe'],
                         ids=['get-failure', 'access-denied', 'body-read-failure', 'invalid-utf8'])
def test_failed_writer_does_not_hide_healthy_writers(logs, monkeypatch, position, failure):
    items = [(key('early'), encoded('a', 'healthy-early')),
             (key('late'), encoded('c', 'healthy-late'))]
    items.insert(position, (key('broken'), failure))
    client = FakeS3(items)
    monkeypatch.setattr(logs, '_s3_client', client)
    (logs.LOG_DIR / ('query_results_' + DATE + '.log')).write_bytes(encoded('b', 'local'))
    assert logs.read_records(DATE) == [
        {'ts': 'a', 'source': 'healthy-early'},
        {'ts': 'b', 'source': 'local'},
        {'ts': 'c', 'source': 'healthy-late'},
    ]
    assert client.read_keys == [item[0] for item in items]

def test_healthy_mirrors_still_deduplicate_local_and_sort(logs, monkeypatch):
    client = FakeS3([(key('late'), encoded('c', 'late')),
                     (key('early'), encoded('a', 'early'))])
    monkeypatch.setattr(logs, '_s3_client', client)
    (logs.LOG_DIR / ('query_results_' + DATE + '.log')).write_bytes(encoded('c', 'late'))
    assert logs.read_records(DATE) == [{'ts': 'a', 'source': 'early'}, {'ts': 'c', 'source': 'late'}]

def test_listing_failure_still_returns_local_records(logs, monkeypatch):
    class FailedList:
        def get_paginator(self, name):
            raise RuntimeError('synthetic listing failure')
    monkeypatch.setattr(logs, '_s3_client', FailedList())
    (logs.LOG_DIR / ('query_results_' + DATE + '.log')).write_bytes(encoded('b', 'local'))
    assert logs.read_records(DATE) == [{'ts': 'b', 'source': 'local'}]

def test_bad_date_never_contacts_s3(logs, monkeypatch):
    class NoS3:
        def get_paginator(self, name):
            raise AssertionError('invalid date must not contact S3')
    monkeypatch.setattr(logs, '_s3_client', NoS3())
    assert logs.read_records('../outside') == []

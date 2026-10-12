"""Offline regressions for normalized protocol links, no API calls."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'protocolnerd-backend'))
from protocolsio_client import _normalize

@pytest.mark.parametrize('doi, expected', [
    ('10.17504/protocols.io.abc123', 'https://doi.org/10.17504/protocols.io.abc123'),
    (' 10.17504/protocols.io.abc123 ', 'https://doi.org/10.17504/protocols.io.abc123'),
    ('doi.org/10.17504/protocols.io.abc123', 'https://doi.org/10.17504/protocols.io.abc123'),
    ('dx.doi.org/10.17504/protocols.io.abc123', 'https://dx.doi.org/10.17504/protocols.io.abc123'),
    ('https://doi.org/10.17504/protocols.io.abc123', 'https://doi.org/10.17504/protocols.io.abc123'),
    ('http://dx.doi.org/10.17504/protocols.io.abc123', 'http://dx.doi.org/10.17504/protocols.io.abc123'),
])
def test_doi_is_resolver_link(doi, expected):
    assert _normalize({'doi': doi, 'url':'https://example.com/old-slug'})['url'] == expected

@pytest.mark.parametrize('record, expected', [
    ({'url':'https://example.com/protocol'}, 'https://example.com/protocol'),
    ({'uri':'published-slug'}, 'https://www.protocols.io/view/published-slug'),
    ({}, ''),
])
def test_no_doi_keeps_existing_fallback(record, expected):
    assert _normalize(record)['url'] == expected

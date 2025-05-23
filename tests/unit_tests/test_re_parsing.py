import re
import pytest

@ pytest.mark.parametrize("l", [
    '@bundle()\njklasjdflksd',
    '@bundle\\ ajsdkfldsjf',
    '@.....bundle(jkalsdfj',
    '@.....bundle\\jklasjdlfk',
])
def test_bundle_decorator_patterns(l):
    # Matches literal @bundle( or @bundle\\  or any @... .bundle(... or @... .bundle\\...
    assert (
        '@bundle(' in l
        or '@bundle\\' in l
        or re.search(r'@.*\.bundle\(.*', l) is not None
        or re.search(r'@.*\.bundle\\.*', l) is not None
    )
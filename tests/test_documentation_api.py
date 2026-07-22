from __future__ import annotations

import re
from pathlib import Path

import bindcurve

API_DOCS_DIR = Path(__file__).parents[1] / "docs" / "api"
AUTODOC_OBJECT = re.compile(
    r"^\.\. auto(?:class|data|function)::\s+bindcurve\.([A-Za-z_]\w*)$",
    re.MULTILINE,
)


def test_api_reference_documents_exactly_the_public_root_api():
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(API_DOCS_DIR.glob("*.md"))
    )
    documented = set(AUTODOC_OBJECT.findall(sources))

    assert documented == set(bindcurve.__all__)

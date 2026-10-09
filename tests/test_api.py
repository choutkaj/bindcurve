import re
from pathlib import Path

import bindcurve as bc

API_DOCS = Path(__file__).parents[1] / "docs" / "api"
AUTODOC = re.compile(
    r"^\.\. auto(?:class|data|function)::\s+bindcurve\.(\w+)$", re.MULTILINE
)


def test_api_reference_documents_exactly_the_public_api():
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in API_DOCS.glob("*.md")
    )
    assert set(AUTODOC.findall(sources)) == set(bc.__all__)

"""tools/mutation_screen.py: deterministic decoders and style routing."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import mutation_screen as ms                                    # noqa: E402


def test_decoders_invert_their_ciphers():
    assert ms.DECODERS["caesar"]("Khoor, Zruog!") == "Hello, World!"
    assert ms.DECODERS["atbash"]("Svool") == "Hello"
    assert ms.DECODERS["morse"](".... .. / - .... . .-. .") == "hi there"
    assert ms.DECODERS["ascii"]("72 105 33") == "Hi!"
    assert ms.DECODERS["ascii"]("no digits here") == "no digits here"


def test_style_kinds():
    assert ms.kind("caesar") == "cipher" and ms.kind("Morse") == "cipher"
    assert ms.kind("translate-fr") == "translate"
    assert ms.kind("role_play") == "english"

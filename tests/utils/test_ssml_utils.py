"""
Regression coverage for strip_ssml_tags' trailing-fragment handling.

A streaming TTS chunk boundary can split an SSML tag anywhere inside it, so
the leaked remainder at the start of the next chunk isn't always a full
key="value"> shape — it can be just the value+quote, or a bare >/>. The
function must strip all of these while leaving legitimate spoken text
containing a literal ">" untouched.
"""
from app.utils.ssml_utils import strip_ssml_tags


def test_strip_ssml_tags_leaves_legitimate_gt_in_sentence_untouched():
    assert strip_ssml_tags("items > 5 in stock") == "items > 5 in stock"


def test_strip_ssml_tags_strips_bare_leading_gt_fragment():
    # Chunk boundary landed exactly at the '>' — nothing of the attribute
    # value carried over.
    assert strip_ssml_tags(">Hello there") == "Hello there"


def test_strip_ssml_tags_strips_value_and_quote_only_fragment():
    # Chunk boundary landed inside the attribute value, missing the
    # leading `key="`.
    assert strip_ssml_tags('90%">Hello') == "Hello"


def test_strip_ssml_tags_strips_full_key_value_fragment():
    assert strip_ssml_tags('rate="90%">Hello') == "Hello"


def test_strip_ssml_tags_strips_self_closing_fragment():
    assert strip_ssml_tags('time="400ms"/>Absolutely!') == "Absolutely!"


def test_strip_ssml_tags_strips_fragment_with_leading_space():
    assert strip_ssml_tags(' pitch="+1st">how are you?') == "how are you?"

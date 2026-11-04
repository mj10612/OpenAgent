import pytest

from openagent.tools.agent_tools import _HTMLTextExtractor


@pytest.mark.parametrize(('source', 'expected'), [
    ('<p>Write &amp;lt;script&amp;gt; literally.</p>', 'Write &lt;script&gt; literally.'),
    ('<p>&amp;amp; &amp;#65; &amp;#x41;</p>', '&amp; &#65; &#x41;'),
    ('<p>&amp;amp;amp;</p>', '&amp;amp;'),
    ('<p>A &amp; B &lt; C &gt; D &quot;ok&quot;</p>', 'A & B < C > D "ok"'),
    ('<p>&#65; &#x41; &#38;</p>', 'A A &'),
    ('<p>ordinary text</p>', 'ordinary text'),
    ('<p> first </p><div> second<br> third </div>', 'first\nsecond\nthird'),
    ('<script>ignored &amp;lt;</script><style>ignored</style><p>kept &amp;lt;</p>', 'kept &lt;'),
    ('<noscript><div>ignored</div>ignored</noscript><p>visible</p>', 'visible'),
])
def test_html_entities_decoded_once(source, expected):
    parser = _HTMLTextExtractor()
    parser.feed(source)
    parser.close()
    assert parser.get_text() == expected
    assert parser.get_text() == expected

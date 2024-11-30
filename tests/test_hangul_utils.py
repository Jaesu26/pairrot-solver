from pairrot.hangul.utils import decompose_hangul, extract_chosung, extract_jongsung, extract_jungsung


def test_extract_chosung():
    assert extract_chosung("강") == "ㄱ"
    assert extract_chosung("안") == "ㅇ"


def test_extract_jungsung():
    assert extract_jungsung("강") == "ㅏ"
    assert extract_jungsung("안") == "ㅏ"


def test_extract_jongsung():
    assert extract_jongsung("강") == "ㅇ"
    assert extract_jongsung("안") == "ㄴ"


def test_decompose_hangul():
    assert decompose_hangul("갉") == ("ㄱ", "ㅏ", "ㄹ", "ㄱ")
    assert decompose_hangul("광") == ("ㄱ", "ㅗ", "ㅏ", "ㅇ")

from pairrot.utils import extract_chosung, extract_jungsung, extract_jongsung


def test_extract_chosung():
    assert extract_chosung("강") == "ㄱ"
    assert extract_chosung("안") == "ㅇ"


def test_extract_jungsung():
    assert extract_jungsung("강") == "ㅏ"
    assert extract_jungsung("안") == "ㅏ"


def test_extract_jongsung():
    assert extract_jongsung("강") == "ㅇ"
    assert extract_jongsung("안") == "ㄴ"

from pairrot.hints import Apple, Banana, Eggplant, Garlic, Mushroom, Carrot


def test_apple():
    apple = Apple("맑")
    assert not apple.can_be_answer(syllable_direct="고", syllable_indirect="수")
    assert apple.can_be_answer(syllable_direct="보", syllable_indirect="법")


def test_banana():
    banana = Banana("맑")
    assert not banana.can_be_answer(syllable_direct="고", syllable_indirect="명")
    assert banana.can_be_answer(syllable_direct="보", syllable_indirect="물")


def test_eggplant():
    eggplant = Eggplant("맑")
    assert not eggplant.can_be_answer(syllable_direct="망", syllable_indirect="상")
    assert eggplant.can_be_answer(syllable_direct="미", syllable_indirect="숙")


def test_garlic():
    garlic = Garlic("맑")
    assert not garlic.can_be_answer(syllable_direct="망", syllable_indirect="상")
    assert garlic.can_be_answer(syllable_direct="악", syllable_indirect="마")


def test_mushroom():
    mushroom = Mushroom("맑")
    assert not mushroom.can_be_answer(syllable_direct="강", syllable_indirect="수")
    assert mushroom.can_be_answer(syllable_direct="막", syllable_indirect="상")


def test_carrot():
    carrot = Carrot("맑")
    assert not carrot.can_be_answer(syllable_direct="막", syllable_indirect="막")
    assert carrot.can_be_answer(syllable_direct="맑", syllable_indirect="음")

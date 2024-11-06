from pairrot.hints import Apple, Banana, Carrot, Eggplant, Garlic, Mushroom


def test_apple():
    apple = Apple("맑", position="first")
    assert not apple.can_be_answer("고수")
    assert apple.can_be_answer("보법")


def test_banana():
    banana = Banana("맑", position="first")
    assert not banana.can_be_answer("고명")
    assert banana.can_be_answer("보물")


def test_eggplant():
    eggplant = Eggplant("맑", position="first")
    assert not eggplant.can_be_answer("망상")
    assert eggplant.can_be_answer("미숙")


def test_garlic():
    garlic = Garlic("맑", position="first")
    assert not garlic.can_be_answer("망상")
    assert garlic.can_be_answer("악마")


def test_mushroom():
    mushroom = Mushroom("맑", position="first")
    assert not mushroom.can_be_answer("강수")
    assert mushroom.can_be_answer("막상")


def test_carrot():
    carrot = Carrot("맑", position="first")
    assert not carrot.can_be_answer("막막")
    assert carrot.can_be_answer("맑음")

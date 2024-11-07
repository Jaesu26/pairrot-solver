from pairrot.hints import Apple, Banana, Carrot, Eggplant, Garlic, Mushroom


def test_apple():
    apple = Apple("맑", position="first")
    assert not apple.is_compatible("고수")
    assert apple.is_compatible("보법")


def test_banana():
    banana = Banana("맑", position="first")
    assert not banana.is_compatible("고명")
    assert banana.is_compatible("보물")


def test_eggplant():
    eggplant = Eggplant("맑", position="first")
    assert not eggplant.is_compatible("망상")
    assert eggplant.is_compatible("미숙")


def test_garlic():
    garlic = Garlic("맑", position="first")
    assert not garlic.is_compatible("망상")
    assert garlic.is_compatible("악마")


def test_mushroom():
    mushroom = Mushroom("맑", position="first")
    assert not mushroom.is_compatible("강수")
    assert mushroom.is_compatible("막상")


def test_carrot():
    carrot = Carrot("맑", position="first")
    assert not carrot.is_compatible("막막")
    assert carrot.is_compatible("맑음")

from core.adversarial import preprocess_text, detect_adversarial_samples

def test_preprocess_text():
    text = "Th!s i$ an exampl3 t3xt."
    processed_text = preprocess_text(text)
    assert processed_text == "this is an example text"

def test_detect_adversarial_samples():
    text = "Th1s t3xt is adv3rsarial."
    is_adversarial = detect_adversarial_samples(text)
    assert is_adversarial is True

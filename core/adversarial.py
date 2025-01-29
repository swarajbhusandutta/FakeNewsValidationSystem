import re
from collections import Counter

def preprocess_text(text):
    """
    Normalize and clean text to defend against common adversarial attacks.
    """
    # Step 1: Lowercase the text
    text = text.lower()

    # Step 2: Remove excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 3: Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Step 4: Replace repeated characters (e.g., "Haaaappy" -> "Happy")
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    return text


def detect_adversarial_patterns(text):
    """
    Detect adversarial patterns in the text and flag if suspicious behavior is observed.
    """
    flags = []

    # 1. Check for excessive use of special characters (even after preprocessing)
    special_char_count = len(re.findall(r'[^\w\s]', text))
    if special_char_count > 0.3 * len(text.split()):
        flags.append("Excessive special character usage.")

    # 2. Check for keyword stuffing (e.g., repeated words)
    word_counts = Counter(text.split())
    repeated_words = [word for word, count in word_counts.items() if count > 5]
    if repeated_words:
        flags.append(f"Keyword stuffing detected for words: {', '.join(repeated_words)}.")

    # 3. Check for unusually short or long content
    if len(text.split()) < 5:
        flags.append("Text is unusually short, may be adversarial.")
    if len(text.split()) > 500:
        flags.append("Text is excessively long, may be adversarial.")

    # 4. Check for obfuscation (e.g., alternating cases or unusual patterns)
    obfuscated_patterns = re.findall(r'[a-z][A-Z]|[A-Z][a-z]', text)
    if len(obfuscated_patterns) > 0.1 * len(text.split()):
        flags.append("Obfuscation detected (e.g., alternating cases).")

    return flags


def detect_adversarial_samples(text):
    """
    Detect adversarial patterns and return a detailed report.
    """
    # Preprocess the input text to normalize it
    cleaned_text = preprocess_text(text)

    # Detect adversarial patterns in the cleaned text
    adversarial_flags = detect_adversarial_patterns(cleaned_text)

    # Return a flag if any patterns are detected, along with the cleaned text
    return {
        "is_adversarial": len(adversarial_flags) > 0,
        "adversarial_flags": adversarial_flags,
        "cleaned_text": cleaned_text
    }

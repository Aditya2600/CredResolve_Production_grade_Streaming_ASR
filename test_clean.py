def clean_qwen_transcript(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    
    import re
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        
    text = re.sub(r"<[^>]+>", "", text)
    
    text = re.sub(r"(?i)\blanguage\s*:\s*\w+\b", "", text)
    text = re.sub(r"(?i)\blanguage\s+\w+\b", "", text)
    
    text = text.strip()
    while len(text) >= 2 and (
        (text[0] in ['"', "'", '“', '‘']) and 
        (text[-1] in ['"', "'", '”', '’'])
    ):
        text = text[1:-1].strip()
        
    return text.strip()

print("1:", repr(clean_qwen_transcript('हाँ बोल।')))
print("2:", repr(clean_qwen_transcript('"हाँ बोल।"')))
print("3:", repr(clean_qwen_transcript('" हाँ बोल।"')))
print("4:", repr(clean_qwen_transcript("'हाँ बोल।'")))
print("5:", repr(clean_qwen_transcript('“हाँ बोल।”')))
print("6:", repr(clean_qwen_transcript('“हाँ बोल।"')))
print("7:", repr(clean_qwen_transcript('हाँ बोल।')))

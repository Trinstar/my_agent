
def clean_text(s: str) -> str:
    return s.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
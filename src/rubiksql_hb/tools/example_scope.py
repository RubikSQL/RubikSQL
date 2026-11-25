"""Validated example contracts and conservative, question-bound applicability."""
import re
_SCOPE_FIELDS = ("answer_grain", "aggregation", "time_scope", "output_fields", "bindings")
_ORDINALS = {word: n for n, word in enumerate(
    ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
     "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth"), 1)}
for _tens, _cardinal, _ordinal in ((20, "twenty", "twentieth"), (30, "thirty", "thirtieth"),
                                  (40, "forty", "fortieth"), (50, "fifty", "fiftieth"),
                                  (60, "sixty", "sixtieth"), (70, "seventy", "seventieth"),
                                  (80, "eighty", "eightieth"), (90, "ninety", "ninetieth")):
    _ORDINALS[_ordinal] = _tens
    _ORDINALS.update({f"{_cardinal}{separator}{word}": _tens + n
                      for word, n in tuple(_ORDINALS.items()) if n < 10 for separator in ("-", " ")})
_ORDINAL = "|".join(re.escape(word) for word in sorted(_ORDINALS, key=len, reverse=True)) + r"|[1-9][0-9]?(?:st|nd|rd|th)"


def _requested_slots(request: str, subject: str) -> tuple[set[int], str]:
    """Read explicit ordinal lists and ranges sharing one field-family subject."""
    term = rf"(?:{_ORDINAL})"
    separator = r"(?:,\s*(?:(?:and|or)\s+)?|(?:and|or|through|to)\s+|[-–]\s*)(?:the\s+)?"
    phrase = rf"\b{term}(?:\s*{separator}{term})*\s+{re.escape(subject)}s?\b"
    def number(word: str) -> int:
        return _ORDINALS[word] if word in _ORDINALS else int(re.sub(r"\D", "", word))
    requested = set()
    for match in re.finditer(phrase, request):
        text = match.group()
        requested.update(number(word) for word in re.findall(rf"\b({_ORDINAL})\b", text))
        for start, end in re.findall(rf"\b({_ORDINAL})\s*(?:through|to|[-–])\s*(?:the\s+)?({_ORDINAL})\b", text):
            a, b = number(start), number(end)
            requested.update(range(min(a, b), max(a, b) + 1))
    return requested, re.sub(phrase, " ", request)


def example_applicable(row: dict, original_question: str) -> bool:
    """Withhold rejected pairs and proven narrower numbered-field precedents.

    Unknown and legacy scopes remain available. A plural/all request is broader than
    a verified numbered-field slice; an explicit ordinal request may use that slice.
    No model call, dev answer, retrieval-query rewrite or database identity is used.

    Args:
        row: Stored example including its independent semantic contract.
        original_question: Runtime-owned complete question and hint.

    Returns:
        bool: Whether the example may be served for this request.
    """
    contract = row.get("contract") or {}
    if contract.get("verdict") == "invalid":
        return False
    text = original_question.lower()
    request, _, hint = text.partition("\n[hint]:")
    for scope in contract.get("field_scopes", []):
        subject, slots = scope["subject"], scope["slots"]
        # Descriptive hints can mention unused slots. Only the request itself
        # establishes requested ordinals; a hint must explicitly bind a meaning.
        requested, unscoped = _requested_slots(request, subject)
        bound = set()
        for column in scope.get("columns", []):
            family = re.sub(r"[1-9][0-9]?$", "", column.lower())
            if family == column.lower():
                continue
            numbered = rf"(?<!\w){re.escape(family)}([1-9][0-9]?)(?!\w)"
            requested.update(int(n) for n in re.findall(numbered, request))
            # A bare schema explanation is not a binding. Read the complete
            # binding clause, including multiple columns, before permitting it.
            for clause in re.split(r"[.;\n]", hint):
                binding = re.split(r"\b(?:means?|refers? to|denotes?|defined as)\b", clause, maxsplit=1)
                if (len(binding) == 2
                        and re.search(rf"\b{re.escape(subject)}s?\b", binding[0])
                        and re.match(rf"\s*[`\"\[]?{numbered}", binding[1])):
                    bound.update(int(n) for n in re.findall(numbered, binding[1]))
        if not requested.issubset(slots):
            return False
        universal = rf"\b(?:all|every|each|both)\s+(?:(?!(?:{_ORDINAL})\b)\w+\s+){{0,2}}{re.escape(subject)}s?\b"
        if re.search(universal, request):
            return False
        if requested:
            if re.search(rf"\b{re.escape(subject)}s\b", unscoped):
                return False
            continue
        if not bound.issubset(slots):
            return False
        if bound:
            continue
        if (re.search(rf"\b{re.escape(subject)}s\b", request)
                or (re.search(rf"\b(?:[2-9]|[1-9][0-9]+|two|three|four|five|six|seven|eight|nine)\s+{re.escape(subject)}s\b", hint)
                    and re.search(r"\b(?:their|all|each|every|both)\b", request))):
            return False
    return True


def example_scope_note(row: dict) -> dict:
    """Return scope as attributed precedent, never as a current-question instruction."""
    contract = row.get("contract") or {}
    return {key: contract[key] for key in (*_SCOPE_FIELDS, "field_scopes") if contract.get(key)}

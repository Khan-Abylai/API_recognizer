import re
import difflib

PATTERNS = {
    "99999999999":    "id",
    "9999999999":     "inn",
    "999999999999":   "iin",
}

MONTHS = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря"
]

DATE_REGEX = re.compile(
    r"^(\d{1,2})([A-Za-zА-Яа-я]+)(\d{4})(?:г(?:ода?)?)?$",
    re.IGNORECASE
)

def build_regex(mask: str) -> str:
    regex_parts = []
    i = 0
    while i < len(mask):
        ch = mask[i]
        if ch == '9':
            start = i
            while i < len(mask) and mask[i] == '9':
                i += 1
            count = i - start
            regex_parts.append(rf"\d{{{count}}}")
        elif ch == 'A':
            start = i
            while i < len(mask) and mask[i] == 'A':
                i += 1
            count = i - start
            regex_parts.append(rf"[A-Za-zА-Яа-я]{{{count}}}")
        else:
            special_chars = ".^$*+?{}[]\\|()"
            if ch in special_chars:
                regex_parts.append("\\" + ch)
            else:
                regex_parts.append(ch)
            i += 1

    regex_str = "".join(regex_parts)
    return "^" + regex_str + "$"


_COMPILED = []
for mask_str, pattern_type in PATTERNS.items():
    pattern_regex = build_regex(mask_str)
    _COMPILED.append(
        (re.compile(pattern_regex), pattern_type)
    )


def parse_and_fix_month(label: str) -> str or None:
    text = label.strip().lower()

    if re.search(r"[a-zа-я]", text, re.IGNORECASE):
        pattern_text_date = re.compile(
            r"^(\d{1,2})([A-Za-zА-Яа-я]+)(\d{4})(?:г(?:ода?)?)?$",
            re.IGNORECASE
        )
        match = pattern_text_date.match(text)
        if not match:
            return None

        day_str, raw_month, year_str = match.group(1), match.group(2), match.group(3)
        try:
            day = int(day_str)
            year = int(year_str)
        except ValueError:
            return None

        best = difflib.get_close_matches(raw_month, MONTHS, n=1, cutoff=0.4)
        if not best:
            return None

        correct_month = best[0]
        return f"{day} {correct_month} {year} года"
    else:
        if len(text) != 8 or not text.isdigit():
            return None

        day_str  = text[0:2]
        month_str = text[2:4]
        year_str = text[4:8]

        try:
            day   = int(day_str)   # '01' -> 1
            month = int(month_str)
            year  = int(year_str)
        except ValueError:
            return None

        if not (1 <= day <= 31):
            return None
        if not (1 <= month <= 12):
            return None

        return f"{day}.{month}.{year}"


def check_label(label: str) -> str:
    for (rgx_compiled, doc_type) in _COMPILED:
        if rgx_compiled.match(label):
            return doc_type, label

    fixed_label = parse_and_fix_month(label)
    if fixed_label is not None:
        return "birthdate", fixed_label

    return None, label

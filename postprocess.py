"""Conservative, offline post-processing for Russian passport OCR."""
import json, re, sys, datetime, csv, os
from difflib import SequenceMatcher

FIELDS = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
    "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
    "Issue_organization_ru",
]
AS_OF = datetime.date(2026, 10, 1)
MIN_CHAR_CONF = 0.985
MIN_FIELD_CONF = 0.98
MIN_BIRTH_PLACE_CHAR_CONF = 0.99
MIN_ISSUER_CHAR_CONF = 0.99
MIN_ISSUER_REFERENCE_SIMILARITY = 0.90

def norm(s):
    return " ".join(str(s or "").split()).upper()

def extract(viz, field):
    v = viz.get(field)
    if not v:
        return "", 0.0, []
    lines = v.get("lines") or []
    value = " ".join(str(line.get("text", "")).strip() for line in lines).strip()
    chars = [float(c) for line in lines for c in (line.get("char_conf") or [])]
    return value, float(v.get("conf") or 0), chars

def parse_date(s):
    try:
        return datetime.datetime.strptime(s, "%d.%m.%Y").date()
    except (ValueError, TypeError):
        return None

def birthday_in_year(birth, year):
    try:
        return birth.replace(year=year)
    except ValueError:
        return datetime.date(year, 2, 28)

def valid_passport(fields):
    """Conservative validity check under standard Russian internal passport age rules."""
    birth = parse_date(fields.get("Birth_date"))
    issued = parse_date(fields.get("Issue_date"))
    if not birth or not issued or issued > AS_OF or birth > AS_OF or issued < birth:
        return False
    age_at_issue = issued.year - birth.year - ((issued.month, issued.day) < (birth.month, birth.day))
    if age_at_issue < 14:
        return False
    if age_at_issue < 20:
        expiry = birthday_in_year(birth, birth.year + 20)
        return AS_OF < expiry
    if age_at_issue < 45:
        expiry = birthday_in_year(birth, birth.year + 45)
        return AS_OF < expiry
    return True

def structurally_plausible(f):
    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", f["Birth_date"]): return False
    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", f["Issue_date"]): return False
    if f["Sex_ru"] not in ("МУЖ.", "ЖЕН."): return False
    if not re.fullmatch(r"\d{3}-\d{3}", f["Issue_organisation_code"]): return False
    if not re.fullmatch(r"\d{2}\s\d{2}\s\d{6}", f["Licence_number"]): return False
    for key in ("Last_name_ru", "First_name_ru"):
        if not re.fullmatch(r"[А-ЯЁ][А-ЯЁ-]*", f[key]): return False
    if not f["Middle_name_ru"] or len(f["Middle_name_ru"]) < 3: return False
    if len(f["Birth_place_ru"]) < 5 or len(f["Issue_organization_ru"]) < 10: return False
    return True

def mrz_check_digit(value):
    """ICAO 9303 check digit (weights 7, 3, 1)."""
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(value):
        if ch.isdigit():
            number = int(ch)
        elif ch == "<":
            number = 0
        elif "A" <= ch <= "Z":
            number = ord(ch) - ord("A") + 10
        else:
            return None
        total += number * weights[i % 3]
    return total % 10


def mrz_consistent(doc, fields):
    """Validate MRZ check digits and cross-check fields visible in both zones."""
    mrz = doc.get("mrz") or {}
    lines = [str(x.get("text", "")).replace(" ", "").upper()
             for x in (mrz.get("lines") or [])]
    line = next((x for x in lines if len(x) >= 41 and x[10:13] == "RUS"), None)
    if not line:
        return False

    if not (line[:9].isdigit() and line[9].isdigit()):
        return False
    if mrz_check_digit(line[:9]) != int(line[9]):
        return False
    if not (line[13:19].isdigit() and line[19].isdigit()):
        return False
    if mrz_check_digit(line[13:19]) != int(line[19]):
        return False

    birth = parse_date(fields["Birth_date"])
    issue = parse_date(fields["Issue_date"])
    if not birth or not issue or birth.year < 1900:
        return False
    if line[13:19] != birth.strftime("%y%m%d"):
        return False
    if line[29:35] != issue.strftime("%y%m%d"):
        return False

    mrz_sex = line[20]
    if (fields["Sex_ru"] == "МУЖ." and mrz_sex != "M") or \
       (fields["Sex_ru"] == "ЖЕН." and mrz_sex != "F"):
        return False

    number = re.sub(r"\D", "", fields["Licence_number"])
    if len(number) != 10 or number[:3] + number[4:] != line[:9]:
        return False

    code = re.sub(r"\D", "", fields["Issue_organisation_code"])
    if len(code) != 6 or code != line[35:41]:
        return False
    return True


def transliterate_ru(value):
    table = {
        "А":"A","Б":"B","В":"V","Г":"G","Д":"D","Е":"E","Ё":"E","Ж":"ZH",
        "З":"Z","И":"I","Й":"I","К":"K","Л":"L","М":"M","Н":"N","О":"O",
        "П":"P","Р":"R","С":"S","Т":"T","У":"U","Ф":"F","Х":"KH","Ц":"TS",
        "Ч":"CH","Ш":"SH","Щ":"SHCH","Ъ":"","Ы":"Y","Ь":"","Э":"E","Ю":"IU","Я":"IA",
        "-":""
    }
    return "".join(table.get(ch, ch) for ch in norm(value) if ch != " ")

def geometry_complete(doc, field):
    v = (doc.get("viz") or {}).get(field)
    if not v:
        return False
    box = v.get("box") or []
    lines = v.get("lines") or []
    if len(box) >= 4 and lines:
        box_h = float(box[3])
        heights = [float((ln.get("box") or [0,0,0,0])[3]) for ln in lines if len(ln.get("box") or []) >= 4]
        if heights and box_h > 1.55 * max(heights) and len(lines) == 1:
            return False
    return True

def normalize_mrz_name(value):
    """Undo frequent OCR confusions in MRZ name text before cross-checking.

    These substitutions are deliberately narrow: 3 is often read for CH, 4
    for SH, 8 for A, and Q for I. We do not use unrestricted edit distance,
    since that can hide a real disagreement between the two document zones.
    """
    value = str(value or "").upper().replace("<", "")
    for wrong, right in (("3", "CH"), ("4", "SH"), ("8", "A"), ("Q", "I")):
        value = value.replace(wrong, right)
    return value


def mrz_names_consistent(doc, fields):
    mrz = doc.get("mrz")
    if not mrz or not mrz.get("lines"):
        return False
    line = next((str(x.get("text", "")).replace(" ", "").upper()
                 for x in mrz["lines"]
                 if str(x.get("text", "")).upper().startswith("PNRUS") and "<<" in str(x.get("text", ""))), None)
    if not line:
        return False
    try:
        surname, rest = line[5:].split("<<", 1)
        given = rest.split("<", 1)[0]
    except (IndexError, ValueError):
        return False
    return (transliterate_ru(fields["Last_name_ru"]) == normalize_mrz_name(surname)
            and transliterate_ru(fields["First_name_ru"]) == normalize_mrz_name(given))

_SUBDIVISION_CODES = None
_SUBDIVISIONS_BY_CODE = None

def subdivision_names_by_code():
    """Load official/known issuing-authority variants grouped by division code."""
    global _SUBDIVISION_CODES, _SUBDIVISIONS_BY_CODE
    if _SUBDIVISIONS_BY_CODE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "subdivisions.csv")
        grouped = {}
        try:
            with open(path, encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    code = row.get("code", "").strip()
                    name = norm(row.get("name", ""))
                    if code and name:
                        grouped.setdefault(code, set()).add(name)
        except OSError:
            grouped = {}
        _SUBDIVISIONS_BY_CODE = grouped
        _SUBDIVISION_CODES = set(grouped)
    return _SUBDIVISIONS_BY_CODE

def known_subdivision_code(code):
    return code in subdivision_names_by_code()

def issuer_reference_similarity(code, issuer_name):
    """Best string similarity to a known issuing-authority name for this code."""
    options = subdivision_names_by_code().get(code, ())
    if not options or not issuer_name:
        return 0.0
    candidate = norm(issuer_name)
    return max(SequenceMatcher(None, candidate, option).ratio() for option in options)

def process(doc):
    fields, confidences, char_mins = {}, {}, {}
    for key in FIELDS:
        value, conf, chars = extract(doc.get("viz") or {}, key)
        fields[key] = value
        confidences[key] = conf
        char_mins[key] = min(chars) if chars else 0.0
    accept = all(fields[k] for k in FIELDS)
    accept = accept and all(confidences[k] >= MIN_FIELD_CONF for k in FIELDS)
    accept = accept and all(char_mins[k] >= MIN_CHAR_CONF for k in FIELDS)
    accept = accept and char_mins["Birth_place_ru"] >= MIN_BIRTH_PLACE_CHAR_CONF
    accept = accept and char_mins["Issue_organization_ru"] >= MIN_ISSUER_CHAR_CONF
    accept = accept and structurally_plausible(fields)
    accept = accept and known_subdivision_code(fields["Issue_organisation_code"])
    accept = accept and valid_passport(fields)
    accept = accept and mrz_consistent(doc, fields)
    accept = accept and geometry_complete(doc, "Birth_place_ru")
    accept = accept and geometry_complete(doc, "Issue_organization_ru")
    accept = accept and issuer_reference_similarity(
        fields["Issue_organisation_code"], fields["Issue_organization_ru"]
    ) >= MIN_ISSUER_REFERENCE_SIMILARITY
    accept = accept and mrz_names_consistent(doc, fields)
    result = {"doc_id": doc["doc_id"], "decision": "accept" if accept else "reject"}
    if accept:
        result["fields"] = fields
    return result

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python postprocess.py data/test_predictions.jsonl answers.jsonl")
    with open(sys.argv[1], encoding="utf-8") as src, open(sys.argv[2], "w", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                doc = json.loads(line)
                dst.write(json.dumps(process(doc), ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

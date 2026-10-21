"""ISO 3166 country normalization, so location filters can rely on exact codes (e.g. HR for Croatia)."""

import pycountry

# Codes models commonly produce that are not ISO 3166-1 alpha-2.
_CODE_ALIASES = {"UK": "GB", "EL": "GR"}
# User-assigned codes in common use that pycountry doesn't list.
_EXTRA_CODES = {"XK": "Kosovo"}


def normalize_code(code: str) -> str:
    """Upper-cased ISO 3166-1 alpha-2 code; raises ValueError for unknown codes."""
    code = _CODE_ALIASES.get(code.strip().upper(), code.strip().upper())
    if code in _EXTRA_CODES or pycountry.countries.get(alpha_2=code):
        return code
    raise ValueError(f"unknown ISO 3166-1 alpha-2 country code: {code!r}")


def name(code: str) -> str:
    """Short English name for a normalized code, e.g. HR → Croatia, CZ → Czechia."""
    if code in _EXTRA_CODES:
        return _EXTRA_CODES[code]
    country = pycountry.countries.get(alpha_2=code)
    return getattr(country, "common_name", None) or country.name


def code_from_name(country_name: str) -> str | None:
    """Best-effort code for a free-text country name (used to backfill older rows)."""
    for code, extra in _EXTRA_CODES.items():
        if country_name.strip().lower() == extra.lower():
            return code
    try:
        return pycountry.countries.lookup(country_name.strip()).alpha_2
    except LookupError:
        try:
            return pycountry.countries.search_fuzzy(country_name.strip())[0].alpha_2
        except LookupError:
            return None

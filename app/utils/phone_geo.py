"""
Lightweight E.164 phone number geo parsing — no third-party dependency.

Used by the outbound number rotation flow (app.services.batch_call_service) to
find a same-country / same-area-code replacement number from the tenant's pool.
This is a best-effort heuristic, not a full libphonenumber-equivalent parser:
the "area code" is simply the 3 digits immediately following the country
calling code, which is correct for NANP (+1) and a reasonable approximation
for most other national numbering plans.
"""
from __future__ import annotations

from typing import Optional

# ITU-T E.164 country calling codes (digits only, no leading '+'), grouped by length
# so the longest matching prefix is tried first.
_COUNTRY_CALLING_CODES: frozenset[str] = frozenset(
    {
        # 1-digit
        "1", "7",
        # 2-digit
        "20", "27", "30", "31", "32", "33", "34", "36", "39", "40", "41", "43",
        "44", "45", "46", "47", "48", "49", "51", "52", "53", "54", "55", "56",
        "57", "58", "60", "61", "62", "63", "64", "65", "66", "81", "82", "84",
        "86", "90", "91", "92", "93", "94", "95", "98",
        # 3-digit
        "211", "212", "213", "216", "218", "220", "221", "222", "223", "224",
        "225", "226", "227", "228", "229", "230", "231", "232", "233", "234",
        "235", "236", "237", "238", "239", "240", "241", "242", "243", "244",
        "245", "246", "248", "249", "250", "251", "252", "253", "254", "255",
        "256", "257", "258", "260", "261", "262", "263", "264", "265", "266",
        "267", "268", "269", "290", "291", "297", "298", "299", "350", "351",
        "352", "353", "354", "355", "356", "357", "358", "359", "370", "371",
        "372", "373", "374", "375", "376", "377", "378", "379", "380", "381",
        "382", "383", "385", "386", "387", "389", "420", "421", "423", "500",
        "501", "502", "503", "504", "505", "506", "507", "508", "509", "590",
        "591", "592", "593", "594", "595", "596", "597", "598", "599", "670",
        "672", "673", "674", "675", "676", "677", "678", "679", "680", "681",
        "682", "683", "685", "686", "687", "688", "689", "690", "691", "692",
        "850", "852", "853", "855", "856", "880", "886", "960", "961", "962",
        "963", "964", "965", "966", "967", "968", "970", "971", "972", "973",
        "974", "975", "976", "977", "992", "993", "994", "995", "996", "998",
    }
)


def get_country_code(e164_number: str) -> Optional[str]:
    """Return the calling code (e.g. '1', '61') for an E.164 number, or None."""
    digits = e164_number.lstrip("+")
    for length in (3, 2, 1):
        prefix = digits[:length]
        if prefix in _COUNTRY_CALLING_CODES:
            return prefix
    return None


def get_area_code(e164_number: str) -> Optional[str]:
    """Best-effort area code: the 3 digits following the country calling code."""
    country_code = get_country_code(e164_number)
    if country_code is None:
        return None
    digits = e164_number.lstrip("+")
    rest = digits[len(country_code):]
    return rest[:3] if len(rest) >= 3 else None

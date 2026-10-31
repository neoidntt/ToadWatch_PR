# test (v2)

Final testing data (locked, used once for the final result).
**LOCKED.** These files are read-only and the split is fingerprinted. Looking is fine, but never use this set to choose features or settings - it is used once for the final result.

* records: **818** (fraud 409, safe 409); message types: email 274, sms 270, url 274
* seed 2027; label `1` = fraud, `0` = safe (taken from the source folder)

| file | what it is |
|---|---|
| `paired_records.csv` | one row per TOAD case: `call_text` + `msg_text` / `msg_url` + `label`, with sources |
| `calls.csv` | the call transcripts only (`record_id`, `call_id`, `text`, `label`, `source`) |
| `messages.csv` | the SMS / email / URL items only (`kind` = sms / email / url, `lang` = en / es) |

A call and a message are paired only inside the same split and with the same label, so no call or message appears in two splits.

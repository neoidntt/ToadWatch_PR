# test

Final testing data (locked, used once for the final result).
**LOCKED.** These files are read-only and the split is fingerprinted. Looking is fine, but never use this set to choose features or settings - it is used once for the final result.

* records: **925** (fraud 463, safe 462); message types: email 342, sms 475, url 108; tiers: matched 630, unmatched 202, hard 93
* `label` = 1 when the ATTACK is fraud, i.e. when the call OR the message is fraud (`call_label` / `msg_label` are the labels of each half)

| file | what it is |
|---|---|
| `paired_records.csv` | one row per attack: `call_text` + `msg_text` / `msg_url`, `label`, `call_label`, `msg_label`, `pair_type` (FF, FS, SF, SS), `tier` |
| `calls.csv` | the call transcripts only |
| `messages.csv` | the SMS / email / URL items only (`kind` = sms / email / url, `lang` = en / es) |

A call and a message are paired only inside the same split, so no call or message appears in two splits; near-identical scripts
(a "family") are never split between sets.

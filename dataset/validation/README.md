# validation

Fitting / validation data (tune the settings).
* records: **461** (fraud 231, safe 230); message types: email 181, sms 225, url 55; tiers: matched 315, unmatched 101, hard 45
* `label` = 1 when the ATTACK is fraud, i.e. when the call OR the message is fraud (`call_label` / `msg_label` are the labels of each half)

| file | what it is |
|---|---|
| `paired_records.csv` | one row per attack: `call_text` + `msg_text` / `msg_url`, `label`, `call_label`, `msg_label`, `pair_type` (FF, FS, SF, SS), `tier` |
| `calls.csv` | the call transcripts only |
| `messages.csv` | the SMS / email / URL items only (`kind` = sms / email / url, `lang` = en / es) |

A call and a message are paired only inside the same split, so no call or message appears in two splits; near-identical scripts
(a "family") are never split between sets.

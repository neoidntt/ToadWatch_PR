# train (v3)

Training data (fit the models).
* records: **3007** (fraud 1200, safe 1807); message types: email 974, sms 1170, url 863
* seed 2028; label `1` = fraud, `0` = safe (taken from the source folder)

| file | what it is |
|---|---|
| `paired_records.csv` | one row per TOAD case: `call_text` + `msg_text` / `msg_url` + `label`, with sources |
| `calls.csv` | the call transcripts only (`record_id`, `call_id`, `text`, `label`, `source`) |
| `messages.csv` | the SMS / email / URL items only (`kind` = sms / email / url, `lang` = en / es) |

A call and a message are paired only inside the same split and with the same label, so no call or message appears in two splits.

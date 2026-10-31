import pandas as pd

from toadwatch.cleaning import (clean_calls, clean_messages, drop_near_duplicates, extract_urls,
                                host_of, near_duplicate_share, norm_key, strip_speakers)


def test_strip_speakers_removes_tags():
    assert strip_speakers("Suspect: Hello. Innocent: Hi there!") == "Hello. Hi there!"
    assert strip_speakers("caller: a receiver: b agent: c customer: d") == "a b c d"


def test_norm_key_ignores_case_and_punctuation():
    assert norm_key("Press 1, NOW!") == norm_key("press 1 now")


def test_host_of():
    assert host_of("https://www.Example.com:8080/a?b=1") == "example.com"
    assert host_of("example.com/path") == "example.com"
    assert host_of("http://[bad") == ""


def test_extract_urls_trims_punctuation():
    assert extract_urls("go to http://a.com/x, or www.b.org.") == ["http://a.com/x", "www.b.org"]


def test_clean_calls_removes_duplicates_conflicts_and_short():
    df = pd.DataFrame({
        "call_id": list("abcdef"),
        "text": ["Suspect: Hello there my friend, press 1 now", "caller: hello THERE my friend press 1 now",
                 "this text is labelled both ways, really", "This text is labelled both ways, really!",
                 "too short", "x" * 30],
        "label": [1, 1, 0, 1, 0, 0], "source": "s", "lang": "en"})
    out, rep = clean_calls(df)
    assert rep["duplicate_rows"] == 1            # a / b are the same call
    assert rep["conflicting_label_rows"] == 2    # c / d disagree on the label
    assert set(out["call_id"]) == {"a", "f"}     # 'too short' is unusable


def test_clean_calls_drops_non_english():
    df = pd.DataFrame({"call_id": ["a", "b"], "text": ["a normal english call text here"] * 2,
                       "label": 1, "source": "s", "lang": ["en", "zh"]})
    out, _ = clean_calls(df)
    assert list(out["call_id"]) == ["a"]


def test_url_hosts_never_repeat_or_cross_classes():
    df = pd.DataFrame({
        "item_id": list("abcd"), "kind": "url", "label": [0, 0, 1, 1], "source": "s", "lang": "en",
        "text": ["http://same.com/a", "https://www.same.com/b", "http://bad.com", "http://both.com"]})
    extra = pd.DataFrame({"item_id": ["e"], "kind": "url", "label": [1], "source": "s", "lang": "en",
                          "text": ["http://both.com/z"]})
    both_legit = pd.DataFrame({"item_id": ["f"], "kind": "url", "label": [0], "source": "s", "lang": "en",
                               "text": ["http://both.com/y"]})
    out, _ = clean_messages(pd.concat([df, extra, both_legit], ignore_index=True))
    hosts = out["url"].map(host_of)
    assert hosts.is_unique
    assert "both.com" not in set(hosts)          # appeared in both classes -> removed
    assert set(hosts) == {"same.com", "bad.com"}


def test_near_duplicates_removed_but_different_calls_kept():
    base = "we would like to inform you that an order was placed for an Apple iPhone using your Amazon account press 1"
    df = pd.DataFrame({"text": [base, base.replace("iPhone", "iPhone 11 Pro") + " thank you",
                                "hi this is the clinic confirming your appointment tomorrow afternoon please call back"]})
    out = drop_near_duplicates(df, threshold=0.8)
    assert len(out) == 2 and out["text"].iloc[0] == base


def test_near_duplicate_share():
    a = pd.Series(["press 1 to talk to an agent about your amazon order", "your appointment is tomorrow"])
    b = pd.Series(["press 1 to talk to an agent about your amazon order!", "totally different words here"])
    assert near_duplicate_share(a, b) == 0.5

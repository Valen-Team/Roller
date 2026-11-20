from __future__ import annotations

import json
from pathlib import Path

from roller import export_audit
from roller.hf_dataset import Rules, _copy_file


def test_tokens_are_redacted_without_current_environment(tmp_path: Path):
    tokens = [
        "sk-proj-" + "a1" * 25,
        "ghp_" + "A" * 36,
        "hf_" + "a" * 34,
        "AKIA" + "A" * 16,
        "ASIA" + "B" * 16,
        "eyJ" + "a" * 12 + ".eyJ" + "b" * 12 + "." + "c" * 20,
        "github_pat_" + "a" * 60,
    ]
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text(
        json.dumps(
            {
                "text": "\n".join(tokens) + " Bearer " + "z" * 24,
                "url": "https://alice:password@host.test/path",
            }
        )
    )
    _copy_file(source, target, Rules(replace=export_audit.replacements()), "source")
    data = json.loads(target.read_text())
    assert all(token not in target.read_text() for token in tokens)
    assert "password" not in data["url"]
    assert "z" * 24 not in data["text"]


def test_private_keys_in_json_and_multiline_text(tmp_path: Path):
    key = "-----BEGIN RSA PRIVATE KEY-----\nSECRET BODY\n-----END RSA PRIVATE KEY-----"
    for text in (key + "\nkeep this", json.dumps({"key": key, "keep": "yes"})):
        source, target = tmp_path / "source", tmp_path / "target"
        source.write_text(text)
        _copy_file(source, target, Rules(replace=export_audit.replacements()), "source")
        assert "SECRET BODY" not in target.read_text()
        assert "keep" in target.read_text()
        if text.startswith("{"):
            assert json.loads(target.read_text())["keep"] == "yes"


def test_text_naming_a_key_header_keeps_the_rest_of_the_file(tmp_path: Path):
    code = "re.findall(r'-----BEGIN PRIVATE KEY-----(.*?)-----END', raw)"
    text = json.dumps({"code": code}, indent=2) + "\nkeep this\n"
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text(text)
    _copy_file(source, target, Rules(replace=export_audit.replacements()), "source")
    assert target.read_text() == text


def test_report_keeps_content_and_does_not_repeat_values(tmp_path: Path):
    output = tmp_path / "dataset"
    output.mkdir()
    text = "alice@example.org 10.2.3.4 /home/alice/work sk-" + "a" * 24
    (output / "sample.jsonl").write_text(json.dumps({"text": text}) + "\n")
    report = export_audit.write_report(output)
    payload = json.loads(report.read_text())
    assert report.parent == output.parent
    assert payload["counts"] == {
        "email": 1,
        "home_path": 1,
        "private_ip": 1,
        "credential": 1,
    }
    assert "alice" not in report.read_text()
    assert json.loads((output / "sample.jsonl").read_text())["text"] == text


def test_report_counts_replaced_user_names_left_in_words(tmp_path: Path):
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "sample.jsonl").write_text(
        json.dumps({"text": "hostnameAlice uid 1000"})
        + "\n"
        + json.dumps({"text": "/Users/user"})
        + "\n"
    )
    report = export_audit.write_report(output, ["alice"])
    payload = json.loads(report.read_text())
    assert payload["counts"] == {"user_name": 1}
    assert payload["findings"] == [
        {"file": "sample.jsonl", "line": 1, "kind": "user_name"}
    ]
    assert "alice" not in report.read_text().lower()


def test_report_handles_long_image_data_without_an_email(tmp_path: Path):
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "sample.jsonl").write_text(
        json.dumps({"image_url": "data:image/png;base64," + "A" * 100_000})
        + "\n"
        + json.dumps({"text": "alice@example.org"})
        + "\n"
    )
    report = json.loads(export_audit.write_report(output).read_text())
    assert report["counts"] == {"email": 1}
    assert report["findings"] == [{"file": "sample.jsonl", "line": 2, "kind": "email"}]

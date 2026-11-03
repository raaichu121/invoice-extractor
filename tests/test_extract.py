"""End-to-end tests of the CLI with the AI call mocked out (no API key needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOOD = json.loads((ROOT / "sample_extracted.json").read_text(encoding="utf-8"))


def test_full_flow_with_mocked_ai(monkeypatch, capsys):
    monkeypatch.setattr(extract, "extract_with_ai", lambda *a, **k: GOOD)
    code = extract.main([str(ROOT / "invoice.png")])
    out = json.loads(capsys.readouterr().out)
    assert code == extract.EXIT_OK
    assert out["data"]["invoice_number"] == "RT-2024-1052"
    assert out["validation"]["is_valid"] is True


def test_ai_misreads_a_number(monkeypatch, capsys):
    bad = json.loads(json.dumps(GOOD))
    bad["total_amount"] = 118565  # model misread a digit
    monkeypatch.setattr(extract, "extract_with_ai", lambda *a, **k: bad)
    code = extract.main([str(ROOT / "invoice.png")])
    out = json.loads(capsys.readouterr().out)
    assert code == extract.EXIT_INVALID
    assert out["validation"]["is_valid"] is False


def test_missing_image(capsys):
    code = extract.main(["does_not_exist.png"])
    assert code == extract.EXIT_ERROR
    assert "not found" in json.loads(capsys.readouterr().out)["error"]


def test_unsupported_file_type(tmp_path, capsys):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    assert extract.main([str(f)]) == extract.EXIT_ERROR


def test_missing_api_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    code = extract.main([str(ROOT / "invoice.png")])
    assert code == extract.EXIT_ERROR
    assert "GEMINI_API_KEY" in json.loads(capsys.readouterr().out)["error"]


def test_output_file_written(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(extract, "extract_with_ai", lambda *a, **k: GOOD)
    out_file = tmp_path / "out.json"
    extract.main([str(ROOT / "invoice.png"), "-o", str(out_file)])
    assert json.loads(out_file.read_text(encoding="utf-8"))["validation"]["is_valid"]

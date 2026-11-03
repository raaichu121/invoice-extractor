import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation import validate_invoice  # noqa: E402

GOOD = {
    "invoice_number": "RT-2024-1052",
    "invoice_date": "2024-10-14",
    "due_date": "2024-11-13",
    "supplier_name": "Rivertech Supplies Ltd.",
    "items": [
        {"description": "HDPE Pipe 110mm (6m length)", "quantity": 50, "unit": "pcs", "unit_price": 1250, "line_total": 62500},
        {"description": "HDPE Elbow 90 degree 110mm", "quantity": 20, "unit": "pcs", "unit_price": 450, "line_total": 9000},
        {"description": "HDPE Coupling 110mm", "quantity": 30, "unit": "pcs", "unit_price": 300, "line_total": 9000},
        {"description": "Welding Service", "quantity": 10, "unit": "hrs", "unit_price": 1500, "line_total": 15000},
        {"description": "Delivery Charge", "quantity": 1, "unit": "lot", "unit_price": 5000, "line_total": 5000},
    ],
    "subtotal": 100500,
    "vat_rate": 13,
    "vat_amount": 13065,
    "total_amount": 113565,
}


def make(**changes):
    d = copy.deepcopy(GOOD)
    d.update(changes)
    return d


def test_good_invoice_is_valid():
    r = validate_invoice(GOOD)
    assert r["is_valid"], r["errors"]
    assert r["errors"] == []


def test_wrong_total_is_caught():
    r = validate_invoice(make(total_amount=113000))
    assert not r["is_valid"]
    assert any("total amount" in e for e in r["errors"])


def test_line_total_mismatch_is_caught():
    d = copy.deepcopy(GOOD)
    d["items"][0]["line_total"] = 60000
    r = validate_invoice(d)
    assert not r["is_valid"]
    assert any("Item 1" in e for e in r["errors"])


def test_subtotal_mismatch_is_caught():
    r = validate_invoice(make(subtotal=100000))
    assert not r["is_valid"]
    assert any("subtotal" in e.lower() for e in r["errors"])


def test_wrong_vat_is_caught():
    r = validate_invoice(make(vat_amount=10000, total_amount=110500))
    assert not r["is_valid"]
    assert any("VAT" in e for e in r["errors"])


def test_missing_required_fields_reported_not_crashing():
    r = validate_invoice(make(invoice_number=None, supplier_name=""))
    assert not r["is_valid"]
    assert any("invoice_number" in e for e in r["errors"])
    assert any("supplier_name" in e for e in r["errors"])


def test_bad_date_format():
    r = validate_invoice(make(invoice_date="14 October 2024"))
    assert not r["is_valid"]
    assert any("invoice_date" in e for e in r["errors"])


def test_due_date_before_invoice_date():
    r = validate_invoice(make(due_date="2024-09-01"))
    assert any("earlier" in e for e in r["errors"])


def test_no_items():
    r = validate_invoice(make(items=[]))
    assert not r["is_valid"]
    assert any("No line items" in e for e in r["errors"])


def test_missing_line_total_is_only_a_warning():
    d = copy.deepcopy(GOOD)
    d["items"][0]["line_total"] = None
    r = validate_invoice(d)
    assert r["is_valid"]
    assert r["warnings"]


def test_numbers_as_strings_with_commas():
    d = copy.deepcopy(GOOD)
    d["total_amount"] = "113,565.00"
    assert validate_invoice(d)["is_valid"]


def test_non_dict_input():
    assert not validate_invoice(None)["is_valid"]

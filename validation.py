"""Validation of extracted invoice data.

The AI model can misread a digit, so we never trust its output blindly.
Every check below uses Decimal (not float) to avoid rounding surprises with money.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

TOLERANCE = Decimal("0.01")  # allow 1 paisa of rounding difference
REQUIRED_FIELDS = ("invoice_number", "invoice_date", "supplier_name", "total_amount")


def to_decimal(value: Any) -> Decimal | None:
    """Convert a value to Decimal, or return None if missing / not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= TOLERANCE


def validate_invoice(data: dict) -> dict:
    """Run all checks and return {"is_valid": bool, "errors": [...], "warnings": [...]}.

    errors   -> data is missing or contradicts itself (invoice should not be trusted)
    warnings -> something is unusual but not necessarily wrong
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return {"is_valid": False, "errors": ["Extracted data is not a JSON object."], "warnings": []}

    # 1. Required fields -----------------------------------------------------
    for field in REQUIRED_FIELDS:
        if data.get(field) in (None, ""):
            errors.append(f"Missing required field: '{field}'.")

    # 2. Date format ---------------------------------------------------------
    parsed_dates = {}
    for field in ("invoice_date", "due_date"):
        value = data.get(field)
        if value in (None, ""):
            continue
        try:
            parsed_dates[field] = datetime.strptime(str(value), "%Y-%m-%d")
        except ValueError:
            errors.append(f"'{field}' is not a valid YYYY-MM-DD date: {value!r}.")
    if "invoice_date" in parsed_dates and "due_date" in parsed_dates:
        if parsed_dates["due_date"] < parsed_dates["invoice_date"]:
            errors.append("'due_date' is earlier than 'invoice_date'.")

    # 3. Line items ----------------------------------------------------------
    items = data.get("items")
    items_sum = Decimal("0")
    items_sum_reliable = True

    if not isinstance(items, list) or not items:
        errors.append("No line items found.")
        items_sum_reliable = False
    else:
        for i, item in enumerate(items, start=1):
            label = f"Item {i}"
            if not isinstance(item, dict):
                errors.append(f"{label} is not an object.")
                items_sum_reliable = False
                continue
            if not item.get("description"):
                warnings.append(f"{label} has no description.")

            qty = to_decimal(item.get("quantity"))
            price = to_decimal(item.get("unit_price"))
            line_total = to_decimal(item.get("line_total"))

            if qty is None or price is None:
                errors.append(f"{label} is missing a valid quantity or unit price.")
                items_sum_reliable = False
                continue
            if qty <= 0 or price < 0:
                errors.append(f"{label} has a non-positive quantity or negative price.")

            expected = qty * price
            if line_total is None:
                warnings.append(f"{label} has no line total; using quantity x unit price.")
                line_total = expected
            elif not _close(expected, line_total):
                errors.append(
                    f"{label}: quantity x unit price = {expected:,.2f} "
                    f"but line total is {line_total:,.2f}."
                )
            items_sum += line_total

    # 4. Subtotal / VAT / total ---------------------------------------------
    subtotal = to_decimal(data.get("subtotal"))
    vat = to_decimal(data.get("vat_amount"))
    vat_rate = to_decimal(data.get("vat_rate"))
    total = to_decimal(data.get("total_amount"))

    if subtotal is not None and items_sum_reliable and not _close(items_sum, subtotal):
        errors.append(f"Sum of items ({items_sum:,.2f}) does not match subtotal ({subtotal:,.2f}).")

    if subtotal is not None and vat is not None and vat_rate is not None:
        expected_vat = subtotal * vat_rate / Decimal("100")
        if not _close(expected_vat, vat):
            errors.append(
                f"VAT {vat_rate}% of subtotal should be {expected_vat:,.2f} but invoice says {vat:,.2f}."
            )

    if total is not None:
        if subtotal is not None:
            expected_total = subtotal + (vat or Decimal("0"))
            if vat is None:
                warnings.append("No VAT amount found; assuming total = subtotal.")
            if not _close(expected_total, total):
                errors.append(
                    f"Subtotal + VAT ({expected_total:,.2f}) does not match total amount ({total:,.2f})."
                )
        elif items_sum_reliable:
            warnings.append("No subtotal found; compared total against the sum of items instead.")
            if not _close(items_sum + (vat or Decimal("0")), total):
                errors.append(
                    f"Sum of items + VAT ({items_sum + (vat or Decimal('0')):,.2f}) "
                    f"does not match total amount ({total:,.2f})."
                )
    elif data.get("total_amount") not in (None, ""):
        errors.append(f"'total_amount' is not a valid number: {data.get('total_amount')!r}.")

    return {"is_valid": not errors, "errors": errors, "warnings": warnings}


















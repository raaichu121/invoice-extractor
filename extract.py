#!/usr/bin/env python3
"""AI-powered invoice data extraction.

Reads an invoice image, asks an AI model (Gemini) to extract the key fields,
validates the result, and prints structured JSON.

Usage:
    python extract.py invoice.png
    python extract.py invoice.png --output result.json
    python extract.py --validate-json result.json      # no API key needed
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional

# Optimize DNS resolution on Windows where IPv6 AAAA queries may cause connection timeouts
try:
    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_first_getaddrinfo(host, port, family=0, *args, **kwargs):
        if family in (0, socket.AF_UNSPEC):
            try:
                return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)
            except socket.gaierror:
                pass
        return _orig_getaddrinfo(host, port, family, *args, **kwargs)

    socket.getaddrinfo = _ipv4_first_getaddrinfo
except Exception:
    pass

from pydantic import BaseModel, ValidationError

from validation import validate_invoice

try:  # optional: load GEMINI_API_KEY from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "gemini-3.5-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
SUPPORTED_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # keep requests small and fast

# Exit codes: 0 = extracted and valid, 1 = extracted but validation failed, 2 = could not extract
EXIT_OK, EXIT_INVALID, EXIT_ERROR = 0, 1, 2
MAX_ATTEMPTS = 3  # tries per request when the API is busy or rate-limited


class ExtractionError(Exception):
    """Raised for any problem that stops extraction (bad file, API error, ...)."""


# --- Schema --------------------------------------------------------------------
class Item(BaseModel):
    description: Optional[str] = None   # item description
    quantity: Optional[float] = None    # numeric quantity only, without the unit
    unit: Optional[str] = None          # e.g. pcs, hrs, lot
    unit_price: Optional[float] = None  # price per unit
    line_total: Optional[float] = None  # line total as printed


class Invoice(BaseModel):
    invoice_number: Optional[str] = None  # exactly as printed
    invoice_date: Optional[str] = None    # YYYY-MM-DD
    due_date: Optional[str] = None        # YYYY-MM-DD
    supplier_name: Optional[str] = None   # the seller / issuer, NOT the 'Bill To' party
    customer_name: Optional[str] = None   # the company being billed
    currency: Optional[str] = None        # ISO code, e.g. NPR
    items: list[Item] = []                # every line item, in order
    subtotal: Optional[float] = None      # before tax, as printed
    vat_rate: Optional[float] = None      # percentage, e.g. 13 for 13%
    vat_amount: Optional[float] = None    # as printed
    total_amount: Optional[float] = None  # final total, as printed


PROMPT = (
    "Extract the data from this invoice image and return it as JSON.\n"
    "Rules:\n"
    "- Copy numbers exactly as printed. Do NOT calculate or correct anything yourself.\n"
    "- Use plain numbers (no currency symbols, no thousands separators).\n"
    "- Dates must be YYYY-MM-DD.\n"
    "- If a field is not visible or you are unsure, use null. Never guess.\n"
    "- The supplier is the company that issued the invoice, not the one in 'Bill To'."
)


# --- Image + API ---------------------------------------------------------------
def load_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) or raise ExtractionError with a clear message."""
    p = Path(path)
    if not p.is_file():
        raise ExtractionError(f"Image file not found: {path}")
    media_type = mimetypes.guess_type(p.name)[0]
    if media_type not in SUPPORTED_TYPES:
        raise ExtractionError(f"Unsupported file type '{p.suffix}'. Use PNG, JPG, WEBP or GIF.")
    raw = p.read_bytes()
    if not raw:
        raise ExtractionError(f"Image file is empty: {path}")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ExtractionError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB; please resize it.")
    return base64.standard_b64encode(raw).decode("ascii"), media_type


def get_api_key() -> str | None:
    """Look for the key under the usual names (case-insensitive on the first one)."""
    for name, value in os.environ.items():
        if name.upper() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value.strip():
            return value.strip().strip('"').strip("'")
    return None


def extract_with_ai(image_b64: str, media_type: str, model: str) -> dict:
    """Send the image to Gemini and return the extracted fields as a dict."""
    api_key = get_api_key()
    if not api_key:
        raise ExtractionError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set it in your shell."
        )
    from google import genai  # imported here so --validate-json works without the package
    from google.genai import errors, types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Invoice,
        temperature=0,
    )
    contents = [
        types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type=media_type),
        PROMPT,
    ]

    # Try requested model, and try fallback models if model is unavailable/overloaded
    models_to_try = [model]
    for fb in FALLBACK_MODELS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    response = None
    last_error: Exception | None = None

    for m in models_to_try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(model=m, contents=contents, config=config)
                last_error = None
                break
            except errors.APIError as e:
                last_error = e
                code = getattr(e, "code", None)
                if code in (400, 401, 403) and ("API key" in str(e) or code in (401, 403)):
                    raise ExtractionError("Authentication failed: check that your GEMINI_API_KEY is correct.")
                if code == 404:
                    # Model not available in this tier/version, try next fallback
                    break
                if code in (429, 500, 502, 503, 504) and attempt < MAX_ATTEMPTS:
                    time.sleep(2 ** attempt)  # back off: 2s, 4s, ...
                    continue
                if code in (503, 404):
                    # Overloaded or not available, try next model
                    break
                if code == 429:
                    raise ExtractionError("Rate limit / quota reached even after retries. Wait a minute and try again.")
                raise ExtractionError(f"API error {code}: {getattr(e, 'message', e)}")
            except Exception as e:  # network problems, timeouts, anything unexpected
                last_error = e
                if attempt < MAX_ATTEMPTS:
                    time.sleep(2 ** attempt)
                    continue
                break
        if response is not None:
            break

    if response is None:
        if last_error:
            if isinstance(last_error, errors.APIError):
                code = getattr(last_error, "code", None)
                if code == 404:
                    raise ExtractionError(f"Model '{model}' was not found. Set INVOICE_MODEL to a valid Gemini model.")
                raise ExtractionError(f"API error {code}: {getattr(last_error, 'message', last_error)}")
            raise ExtractionError(
                f"Could not complete the request ({type(last_error).__name__}). Check your internet connection."
            )
        raise ExtractionError("Could not retrieve a response from the AI model.")

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ExtractionError("The model returned an empty response (it may have been blocked). Try again.")
    # Safety net: strip markdown fences if the model added them despite the schema.
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return Invoice.model_validate_json(text).model_dump()
    except ValidationError as e:
        raise ExtractionError(f"The model's answer did not match the expected format: {e.error_count()} problem(s).")


def build_result(data: dict, source: str) -> dict:
    return {"source": source, "data": data, "validation": validate_invoice(data)}


# --- CLI -----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract structured data from an invoice image using AI.")
    parser.add_argument("image", nargs="?", default="invoice.png", help="path to the invoice image (default: invoice.png)")
    parser.add_argument("-o", "--output", help="also write the JSON result to this file")
    parser.add_argument("--model", default=os.environ.get("INVOICE_MODEL", DEFAULT_MODEL), help=f"model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--validate-json", metavar="FILE", help="skip the AI step and only validate an existing JSON file")
    args = parser.parse_args(argv)

    try:
        if args.validate_json:
            try:
                data = json.loads(Path(args.validate_json).read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise ExtractionError(f"File not found: {args.validate_json}")
            except json.JSONDecodeError as e:
                raise ExtractionError(f"{args.validate_json} is not valid JSON: {e}")
            if isinstance(data, dict) and "data" in data and "validation" in data:
                data = data["data"]  # accept a full result file too
            result = build_result(data, args.validate_json)
        else:
            image_b64, media_type = load_image(args.image)
            data = extract_with_ai(image_b64, media_type, args.model)
            result = build_result(data, args.image)
    except ExtractionError as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stdout)
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR

    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        try:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        except OSError as e:
            print(f"Warning: could not write {args.output}: {e}", file=sys.stderr)

    if not result["validation"]["is_valid"]:
        print("Validation FAILED, see 'validation.errors' above.", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
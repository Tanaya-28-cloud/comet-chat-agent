"""
Order lookup tool for the Aster & Row support agent.

Design principle: privacy and status-precedence rules are enforced here,
in code, not left to the model to remember from a prompt instruction.
The model never receives the raw order record — only what this module
decides to return. Concretely:

  - `internal.*`, `customer.name/email/shipping_address` are never even
    read into the returned dict — they're dropped before anything is
    handed back, not filtered out later by an LLM that was told not to
    repeat them.
  - When status is `cancelled` or `returned`, carrier/tracking/estimated
    delivery are actively nulled out in the response, even though they
    may still be present (and stale) in the source data. This stops the
    model from ever seeing "shipped via UPS, arriving Aug 16" on an
    order that was cancelled on Aug 9 — the bad data literally isn't in
    its context to misread.
  - Nothing here invents a delivery date. If `estimated_delivery` is
    null, it stays null; the caller decides how to phrase that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ORDERS_PATH = Path(__file__).resolve().parent.parent / "data" / "orders.json"

# Fields the data dictionary marks customer-safe. Anything not in this set
# (customer.*, internal.*) is structurally excluded — we build the response
# by picking these keys out, rather than by deleting unsafe keys from a
# copy of the full record. Allow-listing is the safer default: a new field
# added to orders.json in the future is excluded until someone deliberately
# adds it here, instead of being exposed by default.
CUSTOMER_SAFE_ORDER_FIELDS = {
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}
CUSTOMER_SAFE_ITEM_FIELDS = {"name", "quantity", "final_sale"}

# Statuses where carrier/tracking/ETA fields must be suppressed even if
# present in the source data, since operational systems may retain stale
# values after cancellation/return (see orders-data-dictionary.md).
STALE_DELIVERY_FIELD_STATUSES = {"cancelled", "returned"}
DELIVERY_FIELDS_TO_SUPPRESS_WHEN_STALE = {"carrier", "tracking_number", "estimated_delivery"}

STATUSES_REQUIRING_HANDOFF = {"exception"}

ORDER_ID_RE = re.compile(r"^ORD-\d+$")


@dataclass
class OrderLookupResult:
    found: bool
    order: Optional[dict[str, Any]] = None
    error: Optional[str] = None            # "missing_order_id" | "not_found"
    requires_human_handoff: bool = False
    handoff_reason: Optional[str] = None
    within_cancellation_window: Optional[bool] = None  # only meaningful when status == "pending"

    def to_tool_result(self) -> dict[str, Any]:
        """What actually gets serialized back to the model as the tool result.
        Deliberately excludes nothing beyond what's already been filtered —
        this method exists mainly as an explicit seam for logging/debug
        output, so it's obvious in the observability layer exactly what
        the model received."""
        result: dict[str, Any] = {"found": self.found}
        if self.error:
            result["error"] = self.error
        if self.order is not None:
            result["order"] = self.order
        if self.requires_human_handoff:
            result["requires_human_handoff"] = True
            result["handoff_reason"] = self.handoff_reason
        if self.within_cancellation_window is not None:
            result["within_cancellation_window"] = self.within_cancellation_window
        return result


class OrderStore:
    """Loads orders.json once and serves normalized, privacy-filtered lookups."""

    def __init__(self, orders_path: Path = ORDERS_PATH):
        raw = json.loads(orders_path.read_text(encoding="utf-8"))
        self.snapshot_at: str = raw["dataset_name"] and raw["snapshot_at"]
        self._orders_by_id: dict[str, dict[str, Any]] = {
            o["order_id"]: o for o in raw["orders"]
        }

    def lookup(self, raw_order_id: Optional[str]) -> OrderLookupResult:
        if raw_order_id is None or not raw_order_id.strip():
            return OrderLookupResult(found=False, error="missing_order_id")

        normalized_id = self._normalize_order_id(raw_order_id)
        order = self._orders_by_id.get(normalized_id)
        if order is None:
            return OrderLookupResult(found=False, error="not_found")

        return self._to_result(order)

    @staticmethod
    def _normalize_order_id(raw: str) -> str:
        """Uppercase + strip whitespace and common surrounding punctuation.
        Per the data dictionary: normalize harmless differences, but do not
        guess a substantially different ID — so this only strips characters
        around the ID, it never edits characters within it."""
        cleaned = raw.strip().strip(".,;:!?'\"()[]").upper()
        return cleaned

    def _to_result(self, order: dict[str, Any]) -> OrderLookupResult:
        status = order["status"]

        safe_order = {k: v for k, v in order.items() if k in CUSTOMER_SAFE_ORDER_FIELDS}
        safe_order["items"] = [
            {k: v for k, v in item.items() if k in CUSTOMER_SAFE_ITEM_FIELDS}
            for item in order.get("items", [])
        ]

        if status in STALE_DELIVERY_FIELD_STATUSES:
            for f in DELIVERY_FIELDS_TO_SUPPRESS_WHEN_STALE:
                safe_order[f] = None

        requires_handoff = status in STATUSES_REQUIRING_HANDOFF
        handoff_reason = "order_exception_requires_review" if requires_handoff else None

        within_window = None
        if status == "pending":
            within_window = self._within_cancellation_window(order)

        return OrderLookupResult(
            found=True,
            order=safe_order,
            requires_human_handoff=requires_handoff,
            handoff_reason=handoff_reason,
            within_cancellation_window=within_window,
        )

    def _within_cancellation_window(self, order: dict[str, Any]) -> bool:
        """30-minute cancellation window, measured against the dataset's
        snapshot_at rather than wall-clock time, per orders-data-dictionary.md
        ('Use it as the current time for any deterministic evaluation
        involving the 30-minute cancellation window.')."""
        from datetime import datetime

        placed_at = datetime.fromisoformat(order["placed_at"].replace("Z", "+00:00"))
        snapshot_at = datetime.fromisoformat(self.snapshot_at.replace("Z", "+00:00"))
        elapsed_minutes = (snapshot_at - placed_at).total_seconds() / 60
        return elapsed_minutes <= 30


# ---------------------------------------------------------------------------
# Tool-calling entry point
# ---------------------------------------------------------------------------

# JSON-schema tool definition, ready to hand to an LLM's tool-use API
# (Anthropic and OpenAI both accept this shape with minor wrapping).
LOOKUP_ORDER_TOOL_SCHEMA = {
    "name": "lookup_order",
    "description": (
        "Look up the current status of an Aster & Row order by order ID. "
        "Returns only customer-safe fields. Call this whenever the customer "
        "asks about an order's status, shipping, or delivery — never guess "
        "or answer from general knowledge. If the customer hasn't provided "
        "an order ID, ask for one instead of calling this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The customer-provided order ID, e.g. 'ORD-1007'. Pass it through as given — normalization happens inside the tool.",
            }
        },
        "required": ["order_id"],
    },
}


_store: Optional[OrderStore] = None


def get_store() -> OrderStore:
    global _store
    if _store is None:
        _store = OrderStore()
    return _store


def lookup_order(order_id: Optional[str]) -> dict[str, Any]:
    """The actual function the agent orchestrator calls when the model
    invokes the `lookup_order` tool. Returns a plain dict, ready to be
    JSON-serialized straight into the tool_result message."""
    result = get_store().lookup(order_id)
    return result.to_tool_result()

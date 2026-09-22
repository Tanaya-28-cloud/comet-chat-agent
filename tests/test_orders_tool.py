from pathlib import Path

from src.orders_tool import OrderStore


FIXTURE = Path(__file__).parent / "fixtures" / "orders_test_fixture.json"


def test_missing_order_id():
    store = OrderStore(FIXTURE)
    result = store.lookup(None)

    assert result.found is False
    assert result.error == "missing_order_id"


def test_unknown_order_id():
    store = OrderStore(FIXTURE)
    result = store.lookup("ORD-9999")

    assert result.found is False
    assert result.error == "not_found"


def test_normalizes_order_id():
    store = OrderStore(FIXTURE)
    result = store.lookup("  ord-2001. ")

    assert result.found is True
    assert result.order["order_id"] == "ORD-2001"


def test_customer_private_fields_are_removed():
    store = OrderStore(FIXTURE)
    result = store.lookup("ORD-2001")

    assert result.found is True

    order = result.order
    assert "customer" not in order
    assert "internal" not in order
    assert "email" not in order


def test_cancelled_order_suppresses_stale_delivery_fields():
    store = OrderStore(FIXTURE)
    result = store.lookup("ORD-2003")

    assert result.found is True
    assert result.order["status"] == "cancelled"
    assert result.order["carrier"] is None
    assert result.order["tracking_number"] is None
    assert result.order["estimated_delivery"] is None


def test_exception_order_requires_handoff():
    store = OrderStore(FIXTURE)
    result = store.lookup("ORD-2004")

    assert result.found is True
    assert result.requires_human_handoff is True
    assert result.handoff_reason == "order_exception_requires_review"


def test_pending_order_uses_snapshot_for_cancellation_window():
    store = OrderStore(FIXTURE)
    result = store.lookup("ORD-2001")

    assert result.found is True
    assert result.within_cancellation_window is True
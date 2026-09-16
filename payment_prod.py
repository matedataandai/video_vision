"""
Streamlit app: one-time payments via Square (Payment Links / Checkout API)

Why this approach:
Square (like most processors) doesn't let raw card numbers touch your server -
that would put you in full PCI-DSS scope. Instead, this app calls Square's
Payment Links API to generate a secure, Square-hosted checkout page. The
customer enters their card there; your app just creates the link and then
polls Square to see whether the resulting order was accepted or declined.

Setup
-----
1. pip install streamlit requests
2. Get credentials from https://developer.squareup.com/apps (switch the
   dashboard toggle to "Production")
   - Production Access Token
   - Production Location ID (under "Locations", or GET /v2/locations)
3. Set environment variables:
     SQUARE_ACCESS_TOKEN=EAAA...
     SQUARE_LOCATION_ID=L...
     SQUARE_ENVIRONMENT=production   # or "sandbox" for testing
4. Run: streamlit run square_payment_app.py

Note: this polls Square's API in a loop rather than listening for a webhook.
Simpler to run (single file, single port, nothing to expose publicly), at
the cost of only checking while the loop is actively running in the
browser tab. Fine for low-volume or in-person use; for something you want
running unattended 24/7, a webhook is the more robust option.

Square's Payment Links API has no built-in expiration/timeout field - links
stay valid indefinitely unless you deactivate them yourself. So when the
polling loop gives up after POLL_TIMEOUT_SECONDS with no payment, this app
calls DELETE /v2/online-checkout/payment-links/{id} to invalidate the link,
rather than leaving a stale, still-payable link floating around.
"""

import os
import time
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SQUARE_ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN", "")
SQUARE_LOCATION_ID = os.getenv("SQUARE_LOCATION_ID", "")
SQUARE_ENVIRONMENT = os.getenv("SQUARE_ENVIRONMENT", "production")  # sandbox | production

BASE_URL = (
    "https://connect.squareup.com"
    if SQUARE_ENVIRONMENT == "production"
    else "https://connect.squareupsandbox.com"
)
SQUARE_VERSION = "2024-10-17"  # Square API version header

HEADERS = {
    "Square-Version": SQUARE_VERSION,
    "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
    "Content-Type": "application/json",
}

POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 300  # give up after 5 minutes


# ---------------------------------------------------------------------------
# Square API helpers
# ---------------------------------------------------------------------------
class SquareAPI():
    def __init__(self, amount_cents: int, currency: str, description: str):
        self.amount_cents = amount_cents
        self.currency = currency
        self.description = description
        self.SQUARE_ACCESS_TOKEN = SQUARE_ACCESS_TOKEN
        self.SQUARE_LOCATION_ID = SQUARE_LOCATION_ID
        self.SQUARE_ENVIRONMENT = SQUARE_ENVIRONMENT

    def create_payment_link(self) -> dict:
        """Create a Square-hosted checkout page for a one-time payment."""
        url = f"{BASE_URL}/v2/online-checkout/payment-links"
        payload = {
            "idempotency_key": str(uuid.uuid4()),
            "quick_pay": {
                "name": self.description or "Payment",
                "price_money": {
                    "amount": self.amount_cents,   # smallest currency unit, e.g. cents
                    "currency": self.currency,
                },
                "location_id": self.SQUARE_LOCATION_ID,
            },
        }
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()


    def get_order_state(self,order_id: str) -> tuple[str, list]:
        """Return (state, tenders) for an order. state is e.g. OPEN/COMPLETED/CANCELED."""
        url = f"{BASE_URL}/v2/orders/{order_id}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        order = resp.json()["order"]
        return order.get("state", "UNKNOWN"), order.get("tenders", [])


    def get_tender_outcome(self,order_id: str) -> str | None:
        """If the order has a tender, look up the underlying payment to see if it
        was accepted (COMPLETED) or declined (FAILED/CANCELED)."""
        state, tenders = self.get_order_state(order_id)
        if state == "COMPLETED" and tenders:
            return "ACCEPTED"
        if state == "CANCELED":
            return "DECLINED"
        return None  # still pending


def delete_payment_link(payment_link_id: str) -> None:
    """Invalidate a payment link so it can no longer be used to pay.
    Square's payment links don't expire on their own (there's no timeout
    field on create), so this is how we enforce our own deadline: once we
    stop waiting, we deactivate the link rather than leaving it payable
    indefinitely.
    """
    url = f"{BASE_URL}/v2/online-checkout/payment-links/{payment_link_id}"
    resp = requests.delete(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Square Payment Demo", page_icon="💳")
st.title("💳 One-Time Payment (Square)")

if not SQUARE_ACCESS_TOKEN or not SQUARE_LOCATION_ID:
    st.error(
        "Missing SQUARE_ACCESS_TOKEN or SQUARE_LOCATION_ID environment variables. "
        "Set them before running the app (see the docstring at the top of this file)."
    )
    st.stop()

st.caption(f"Environment: **{SQUARE_ENVIRONMENT}**")

# Keep the created payment link/order across reruns
if "payment_link" not in st.session_state:
    st.session_state.payment_link = None
if "payment_link_id" not in st.session_state:
    st.session_state.payment_link_id = None
if "order_id" not in st.session_state:
    st.session_state.order_id = None
if "outcome" not in st.session_state:
    st.session_state.outcome = None  # None | "ACCEPTED" | "DECLINED" | "TIMEOUT" | "INVALIDATED"

with st.form("payment_form"):
    description = st.text_input("Description", placeholder="e.g. Consulting invoice #1042")
    amount = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f")
    currency = st.selectbox("Currency", ["USD", "AUD", "GBP", "EUR", "CAD"])
    submitted = st.form_submit_button("Create payment link")

if submitted:
    amount_cents = int(round(amount * 100))
    try:
        with st.spinner("Creating checkout link..."):
            squareapi = SquareAPI(amount_cents, currency, description)
            result = squareapi.create_payment_link()
        link = result["payment_link"]
        st.session_state.payment_link = link["url"]
        st.session_state.payment_link_id = link["id"]
        st.session_state.order_id = link["order_id"]
        st.session_state.outcome = None
        st.success("Payment link created.")
    except requests.HTTPError as e:
        st.error(f"Square API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

if st.session_state.payment_link:
    terminal = st.session_state.outcome in ("ACCEPTED", "DECLINED", "INVALIDATED")

    if not terminal or st.session_state.outcome == "ACCEPTED":
        st.markdown("### Checkout")
        st.link_button("Open Square Checkout ↗", st.session_state.payment_link)
        st.code(st.session_state.payment_link, language=None)

    st.markdown("### Payment status")

    if st.session_state.outcome == "ACCEPTED":
        st.success("✅ Payment accepted.")
    elif st.session_state.outcome == "DECLINED":
        st.error("❌ Payment declined / canceled.")
    elif st.session_state.outcome == "INVALIDATED":
        st.warning(
            f"⏱️ No payment within {POLL_TIMEOUT_SECONDS}s — link deactivated. "
            "It will no longer work if opened."
        )

    if st.button("Wait for payment result", disabled=terminal):
        status_box = st.empty()
        elapsed = 0
        outcome = None
        try:
            while elapsed < POLL_TIMEOUT_SECONDS:
                outcome = squareapi.get_tender_outcome(st.session_state.order_id)
                if outcome is not None:
                    break
                status_box.info(f"Waiting for payment... ({elapsed}s elapsed)")
                time.sleep(POLL_INTERVAL_SECONDS)
                elapsed += POLL_INTERVAL_SECONDS

            if outcome is not None:
                status_box.empty()
                st.session_state.outcome = outcome
            else:
                status_box.info("No payment received — deactivating link...")
                try:
                    delete_payment_link(st.session_state.payment_link_id)
                    st.session_state.outcome = "INVALIDATED"
                except requests.HTTPError as e:
                    st.error(
                        f"Timed out, and failed to deactivate the link: "
                        f"{e.response.status_code} - {e.response.text}"
                    )
                status_box.empty()
            st.rerun()
        except requests.HTTPError as e:
            status_box.empty()
            st.error(f"Square API error: {e.response.status_code} - {e.response.text}")

st.divider()
st.caption(
    "\"Wait for payment result\" polls Square every "
    f"{POLL_INTERVAL_SECONDS}s for up to {POLL_TIMEOUT_SECONDS}s while the "
    "button is running (closing the tab stops it). If no payment comes in "
    "within that window, the link is deactivated via the API rather than "
    "left open indefinitely."
)
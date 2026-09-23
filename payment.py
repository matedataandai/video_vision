"""
Streamlit app: one-time payments via Square (Payment Links / Checkout API)

Why this approach:
Square (like most processors) doesn't let raw card numbers touch your server -
that would put you in full PCI-DSS scope. Instead, this app calls Square's
Payment Links API to generate a secure, Square-hosted checkout page. The
customer enters their card there; your app just creates the link and later
checks whether the resulting order was paid.

Setup
-----
1. pip install streamlit requests python-dotenv
2. Get credentials from https://developer.squareup.com/apps
   - Sandbox Access Token (for testing) or Production Access Token
   - Location ID (Sandbox test account comes with one; find it under
     "Locations" in the dashboard, or via GET /v2/locations)
3. Set environment variables (or use a .env file with python-dotenv):
     SQUARE_ACCESS_TOKEN=EAAA...
     SQUARE_LOCATION_ID=L...
     SQUARE_ENVIRONMENT=sandbox      # or "production"
4. Run: streamlit run square_payment_app.py
"""

import os
import time
import uuid
import requests
import streamlit as st
import time
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SQUARE_ACCESS_TOKEN = os.getenv("SQUARE_ACCESS_TOKEN", "")
SQUARE_LOCATION_ID = os.getenv("SQUARE_LOCATION_ID", "")
SQUARE_ENVIRONMENT = os.getenv("SQUARE_ENVIRONMENT", "")  # sandbox | production

BASE_URL = (
    "https://connect.squareupsandbox.com"
    if SQUARE_ENVIRONMENT == "sandbox"
    else "https://connect.squareup.com"
)
SQUARE_VERSION = "2024-10-17"  # Square API version header

HEADERS = {
    "Square-Version": SQUARE_VERSION,
    "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
    "Content-Type": "application/json",
}
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 5  # give up after 5 minutes

# ---------------------------------------------------------------------------
# Square API helpers
# ---------------------------------------------------------------------------

class SquareAPI:
    def __init__(self, amount_cents: int, currency: str, description: str):
        self.amount_cents = amount_cents
        self.currency = currency
        self.description = description

    def create_payment_link(self) -> dict:
        url = f"{BASE_URL}/v2/online-checkout/payment-links"
        payload = {
            "idempotency_key": str(uuid.uuid4()),
            "quick_pay": {
                "name": self.description or "Payment",
                "price_money": {
                    "amount": self.amount_cents,
                    "currency": self.currency,
                },
                "location_id": SQUARE_LOCATION_ID,
            },
        }
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def get_order_state(order_id: str) -> tuple[str, list]:
        url = f"{BASE_URL}/v2/orders/{order_id}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        order = resp.json()["order"]
        return order.get("state", "UNKNOWN"), order.get("tenders", [])

    @staticmethod
    def get_tender_outcome(order_id: str) -> str | None:
        state, tenders = SquareAPI.get_order_state(order_id)
        if tenders:
            return "ACCEPTED"
        if state == "CANCELED":
            return "DECLINED"
        return None

    @staticmethod
    def delete_payment_link(payment_link_id: str) -> None:
        url = f"{BASE_URL}/v2/online-checkout/payment-links/{payment_link_id}"
        resp = requests.delete(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def SquarePaymentUI(amount=10.00, description="Video recording for Court 1 at Tennis Club", currency="AUD"):
    st.title("💳 One-Time Payment (Square)")

    # Keep the created payment link/order across reruns
    # Initialize persistent session state variables
    if "payment_link" not in st.session_state:
        st.session_state.payment_link = None
    if "payment_link_id" not in st.session_state:
        st.session_state.payment_link_id = None
    if "order_id" not in st.session_state:
        st.session_state.order_id = None
    if "outcome" not in st.session_state:
        st.session_state.outcome = None
    if "payment_started" not in st.session_state:
        st.session_state.payment_started = False

    with st.form("payment_form"):
        st.text_input("Description", value=description, disabled=True)
        st.number_input("Amount", value=amount, format="%.2f", disabled=True)
        st.selectbox("Currency", [currency], disabled=True)
        create_clicked = st.form_submit_button("Create payment link")

    if create_clicked:
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
            st.session_state.payment_started = True
            st.success("Payment link created.")
        except requests.HTTPError as e:
            st.error(f"Square API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

        if st.session_state.payment_link and st.session_state.outcome is None:
            st.markdown("### Checkout")
            st.link_button("Open Square Checkout ↗", st.session_state.payment_link)
            st.code(st.session_state.payment_link, language=None)

            st.markdown("### Payment status")

            status_box = st.empty()
            elapsed = 0
            outcome = None
            try:
                while elapsed < POLL_TIMEOUT_SECONDS:
                    outcome = SquareAPI.get_tender_outcome(st.session_state.order_id)
                    if outcome is not None:
                        break
                    status_box.info(f"Waiting for payment... ({elapsed}s elapsed) - You have {POLL_TIMEOUT_SECONDS}s to make payment.")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    elapsed += POLL_INTERVAL_SECONDS
                if outcome is not None:
                    status_box.empty()
                    st.session_state.outcome = outcome
                else:
                    status_box.info("No payment received — deactivating link...")
                    try:
                        SquareAPI.delete_payment_link(st.session_state.payment_link_id)
                        st.session_state.outcome = "INVALIDATED"
                    except requests.HTTPError as e:
                        st.error(
                            f"Timed out, and failed to deactivate the link: "
                            f"{e.response.status_code} - {e.response.text}"
                        )
                    status_box.empty()
            except requests.HTTPError as e:
                status_box.empty()
                st.error(f"Square API error: {e.response.status_code} - {e.response.text}")
        st.session_state.outcome = SquareAPI.get_tender_outcome(st.session_state.order_id)
        if st.session_state.outcome == "ACCEPTED":
            st.success("✅ Payment accepted.")
        elif st.session_state.outcome == "DECLINED":
            st.error("❌ Payment declined / canceled.")
        elif st.session_state.outcome == "INVALIDATED":
            st.warning(
                f"⏱️ No payment within {POLL_TIMEOUT_SECONDS}s — link deactivated. "
                "It will no longer work if opened."
            )

    st.divider()
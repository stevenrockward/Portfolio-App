"""
Streamlit app — Excel logic recreated in Python, fully API-driven.

Features:
- Enter a ticker (e.g. AAPL)
- Fetch live financials from yfinance (income, balance, cashflow, info)
- Recompute ratios (Current, Quick, Gross/Operating/Net margins, ROA, ROE, EPS, BVPS, PE)
- Valuation models:
    - Graham Number
    - Multi-stage DCF (years 1-5 growth, years 6-10 growth, terminal)
    - Simple PE-based fair value (EPS * chosen PE)
- Competitor comparison: enter comma-separated tickers to compare
- Editable DCF inputs in the sidebar
- No Excel file required
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional

st.set_page_config(page_title="Intrinsic Value — Live API", layout="wide")

# ------------------- Utility / Data retrieval -------------------
@st.cache_data(show_spinner=False)
def fetch_yf(ticker: str) -> dict:
    """Fetch financials, balance sheet, cashflow and info for ticker from yfinance.
       Returns a dict with normalized keys and raw dfs."""
    t = yf.Ticker(ticker)
    info = t.info or {}
    # Fetch tables; handle missing gracefully
    try:
        fin = t.financials.copy()
    except Exception:
        fin = pd.DataFrame()
    try:
        bal = t.balance_sheet.copy()
    except Exception:
        bal = pd.DataFrame()
    try:
        cf = t.cashflow.copy()
    except Exception:
        cf = pd.DataFrame()

    def safe_loc(df: pd.DataFrame, label: str) -> Optional[float]:
        if df is None or df.empty:
            return None
        # try common variants
        candidates = [label, label.replace(" ", ""), label.title(), label.upper()]
        for cand in candidates:
            try:
                val = df.loc[cand].iloc[0]
                return float(val) if not pd.isna(val) else None
            except Exception:
                continue
        # try fuzzy match (case insensitive) on index
        try:
            idx = [str(i).strip().lower() for i in df.index]
            if label.strip().lower() in idx:
                pos = idx.index(label.strip().lower())
                val = df.iloc[pos].iloc[0]
                return float(val) if not pd.isna(val) else None
        except Exception:
            pass
        return None

    data = {
        "ticker": ticker.upper(),
        "info": info,
        # price & shares
        "price": info.get("currentPrice") or info.get("regularMarketPrice") or None,
        "shares_outstanding": info.get("sharesOutstanding"),
        "marketCap": info.get("marketCap"),
        # Income (most recent column)
        "total_revenue": safe_loc(fin, "Total Revenue"),
        "cogs": safe_loc(fin, "Cost Of Revenue") or safe_loc(fin, "Cost of Revenue"),
        "operating_income": safe_loc(fin, "Operating Income"),
        "net_income": safe_loc(fin, "Net Income"),
        # Balance
        "total_assets": safe_loc(bal, "Total Assets"),
        "total_liab": safe_loc(bal, "Total Liab") or safe_loc(bal, "Total Liabilities") or safe_loc(bal, "Total liabilities"),
        "total_equity": safe_loc(bal, "Total Stockholder Equity") or safe_loc(bal, "Total Shareholder Equity") or safe_loc(bal, "Stockholders' Equity"),
        "current_assets": safe_loc(bal, "Total Current Assets"),
        "current_liabilities": safe_loc(bal, "Total Current Liabilities"),
        "inventory": safe_loc(bal, "Inventory"),
        "cash_and_eq": safe_loc(bal, "Cash") or safe_loc(bal, "Cash And Cash Equivalents"),
        "long_term_debt": safe_loc(bal, "Long Term Debt") or info.get("longTermDebt"),
        # Cash flow
        "operating_cf": safe_loc(cf, "Total Cash From Operating Activities") or safe_loc(cf, "Net Cash Provided by Operating Activities"),
        "capex": safe_loc(cf, "Capital Expenditures"),
        "free_cash_flow": None,  # computed below if possible
        # raw dfs for inspection
        "financials_df": fin,
        "balance_df": bal,
        "cashflow_df": cf
    }

    # compute simple FCF = operating_cf + capex (capex is usually negative in yfinance)
    try:
        ocf = data["operating_cf"]
        cap = data["capex"]
        if ocf is not None and cap is not None:
            data["free_cash_flow"] = float(ocf) + float(cap)
    except Exception:
        data["free_cash_flow"] = None

    return data

# ------------------- Ratio calculations -------------------
def compute_common_ratios(d: dict) -> dict:
    """Return dictionary of computed ratios similar to the Excel sheet."""
    def F(k):
        v = d.get(k)
        try:
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan

    revenue = F("total_revenue")
    cogs = F("cogs")
    op_income = F("operating_income")
    net_income = F("net_income")
    assets = F("total_assets")
    equity = F("total_equity")
    current_assets = F("current_assets")
    current_liabilities = F("current_liabilities")
    inventory = F("inventory")
    price = F("price")
    shares = F("shares_outstanding")
    eps_info = d.get("info", {}).get("trailingEps")
    eps_calc = None
    if shares and not np.isnan(net_income := F("net_income")):
        try:
            eps_calc = net_income / shares
        except Exception:
            eps_calc = np.nan

    eps = eps_info if eps_info is not None else (eps_calc if eps_calc is not None else np.nan)
    bvps = None
    if equity and shares and shares > 0:
        bvps = equity / shares
    else:
        bvps = d.get("info", {}).get("bookValue") or np.nan

    ratios = {}
    ratios["Price"] = price
    ratios["EPS (info)"] = eps_info
    ratios["EPS (calc from NI)"] = eps_calc
    ratios["EPS (used)"] = eps
    ratios["Book Value per Share (BVPS)"] = bvps
    ratios["Gross Margin"] = (revenue - cogs) / revenue if revenue and not np.isnan(revenue) and cogs and not np.isnan(cogs) else np.nan
    ratios["Operating Margin"] = op_income / revenue if op_income and revenue else np.nan
    ratios["Net Margin"] = net_income / revenue if net_income and revenue else np.nan
    ratios["Current Ratio"] = current_assets / current_liabilities if current_assets and current_liabilities else np.nan
    ratios["Quick Ratio"] = (current_assets - inventory) / current_liabilities if current_assets and inventory is not None and current_liabilities else np.nan
    ratios["ROA"] = net_income / assets if net_income and assets else np.nan
    ratios["ROE"] = net_income / equity if net_income and equity else np.nan
    ratios["P/E (info)"] = d.get("info", {}).get("trailingPE") or np.nan
    if price and eps and not (eps == 0 or np.isnan(eps)):
        ratios["P/E (calc)"] = price / eps
    else:
        ratios["P/E (calc)"] = np.nan
    ratios["Shares Outstanding"] = shares
    ratios["Market Cap (info)"] = d.get("marketCap") or np.nan
    ratios["Free Cash Flow (most recent)"] = d.get("free_cash_flow") or np.nan

    # Net debt = total_debt - cash (where possible)
    total_debt = d.get("long_term_debt") or np.nan
    cash = d.get("cash_and_eq") or np.nan
    if not np.isnan(total_debt) and not np.isnan(cash):
        ratios["Net Debt"] = total_debt - cash
    else:
        ratios["Net Debt"] = np.nan

    return ratios

# ------------------- Valuation models -------------------
def graham_number(eps: float, bvps: float) -> Optional[float]:
    """Graham Number = sqrt(22.5 * EPS * BVPS)"""
    try:
        if eps is None or bvps is None:
            return None
        eps_f = float(eps)
        bvps_f = float(bvps)
        if eps_f <= 0 or bvps_f <= 0:
            return None
        # compute carefully
        inner = 22.5 * eps_f * bvps_f
        return float(np.sqrt(inner))
    except Exception:
        return None

def dcf_value_per_share(
    starting_fcf: float,
    growth1: float,
    growth2: float,
    terminal_growth: float,
    discount_rate: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
    multiply: float = 1.0
) -> Optional[float]:
    """
    Multi-stage DCF:
      - years 1-5 growth1 (annual)
      - years 6-10 growth2 (annual)
      - terminal value after year 10 using Gordon growth
    Return: intrinsic value per share (float) or None if invalid inputs.
    """
    try:
        fcf = float(starting_fcf) * float(multiply)
        r = float(discount_rate)
        tg = float(terminal_growth)
        if r <= tg:
            return None  # invalid: discount must exceed terminal growth
        pv_sum = 0.0
        # years 1..5
        for i in range(1, 6):
            fcf = fcf * (1 + float(growth1))
            pv = fcf / ((1 + r) ** i)
            pv_sum += pv
        # years 6..10
        for i in range(6, 11):
            fcf = fcf * (1 + float(growth2))
            pv = fcf / ((1 + r) ** i)
            pv_sum += pv
        # terminal value at end of year 10 (use last grown fcf as base)
        terminal_value = fcf * (1 + tg) / (r - tg)
        pv_terminal = terminal_value / ((1 + r) ** 10)
        pv_sum += pv_terminal
        enterprise_value = pv_sum
        equity_value = enterprise_value - float(net_debt or 0.0)
        if shares_outstanding and shares_outstanding > 0:
            return equity_value / float(shares_outstanding)
        return None
    except Exception:
        return None

def pe_fair_value(eps: float, target_pe: float) -> Optional[float]:
    try:
        if eps is None:
            return None
        return float(eps) * float(target_pe)
    except Exception:
        return None

# ------------------- UI / App -------------------
st.title("Intrinsic Value Analyzer — Live (no Excel required)")

st.markdown(
    "Type a ticker and the app will fetch live financials (yfinance), recompute ratios, and run valuation models "
    "that replicate the Excel workbook logic (DCF, Graham, multiples)."
)

col1, col2 = st.columns([2, 1])

with col2:
    st.sidebar.header("DCF Inputs (editable)")
    # Default assumptions (editable)
    default_g1 = st.sidebar.number_input("Growth Years 1-5 (decimal)", value=0.10, format="%.4f")
    default_g2 = st.sidebar.number_input("Growth Years 6-10 (decimal)", value=0.05, format="%.4f")
    default_tg = st.sidebar.number_input("Terminal Growth (decimal)", value=0.02, format="%.4f")
    default_disc = st.sidebar.number_input("Discount Rate / WACC (decimal)", value=0.10, format="%.4f")
    default_multiply = st.sidebar.number_input("Scale factor (if API returns thousands)", value=1.0, format="%.0f")
    st.sidebar.markdown("---")
    st.sidebar.header("Competitors")
    competitor_input = st.sidebar.text_input("Peer tickers (comma-separated)", value="")  # optional
    st.sidebar.caption("If you provide peers (e.g. MSFT,GOOGL), the app will fetch and show basic comparison.")

with col1:
    ticker = st.text_input("Ticker (e.g. AAPL)", value="AAPL").upper().strip()
    run_button = st.button("Run analysis")

if run_button and ticker:
    with st.spinner(f"Fetching data for {ticker} from yfinance..."):
        core = fetch_yf(ticker)
    if not core or core.get("financials_df") is None and core.get("balance_df") is None and core.get("cashflow_df") is None and not core.get("info"):
        st.error("No data returned from yfinance for this ticker. Check the ticker symbol or try again.")
    else:
        # Show basic header
        info = core.get("info", {})
        st.subheader(f"{ticker} — {info.get('longName') or info.get('shortName') or ''}")
        st.write(f"Market Price: {core.get('price'):,}" if core.get('price') else "Price: N/A")
        # compute ratios
        ratios = compute_common_ratios(core)
        st.markdown("### Recomputed Ratios")
        ratio_df = pd.DataFrame.from_dict(ratios, orient="index", columns=["Value"])
        # format numeric nicely
        def fmt(x):
            try:
                if pd.isna(x):
                    return "N/A"
                # large ints: show with commas
                if abs(x) >= 1_000_000_000:
                    return f"{x:,.0f}"
                if abs(x) >= 1_000:
                    return f"{x:,.2f}"
                return f"{x:.4f}" if isinstance(x, float) else str(x)
            except Exception:
                return str(x)
        ratio_df["Formatted"] = ratio_df["Value"].apply(fmt)
        st.dataframe(ratio_df[["Formatted"]])

        # Valuation calculations
        st.markdown("### Valuation Models")
        # Graham
        eps_used = ratios.get("EPS (used)") if ratios.get("EPS (used)") is not None else ratios.get("EPS (calc from NI)")
        bvps_used = ratios.get("Book Value per Share (BVPS)")
        graham = graham_number(eps_used, bvps_used)
        st.write("**Graham Number**")
        st.write(f"- EPS used: {eps_used:.6f}" if not pd.isna(eps_used) else "- EPS: N/A")
        st.write(f"- BVPS: {bvps_used:.6f}" if not pd.isna(bvps_used) else "- BVPS: N/A")
        st.metric("Graham intrinsic value (per share)", f"${graham:,.2f}" if graham else "N/A")

        # DCF: base inputs prefilled from API where possible
        starting_fcf_api = core.get("free_cash_flow") or ratios.get("Free Cash Flow (most recent)") or 0.0
        st.write("**Multi-stage DCF**")
        st.write(f"- Starting FCF (most recent): {starting_fcf_api:,}")
        # allow user override
        starting_fcf = st.number_input("Starting Free Cash Flow (override)", value=float(starting_fcf_api or 0.0), format="%.2f")
        g1 = default_g1
        g2 = default_g2
        tg = default_tg
        discount = default_disc
        multiply = default_multiply
        shares = core.get("shares_outstanding") or ratios.get("Shares Outstanding") or st.number_input("Shares outstanding (if yfinance missing)", value=0.0, format="%.0f")
        net_debt_guess = ratios.get("Net Debt") if not pd.isna(ratios.get("Net Debt")) else 0.0
        net_debt = st.number_input("Net Debt (override, negative = net cash)", value=float(net_debt_guess or 0.0), format="%.2f")

        dcf_val = dcf_value_per_share(
            starting_fcf=starting_fcf,
            growth1=g1,
            growth2=g2,
            terminal_growth=tg,
            discount_rate=discount,
            shares_outstanding=shares,
            net_debt=net_debt,
            multiply=multiply
        )
        st.metric("DCF intrinsic value (per share)", f"${dcf_val:,.2f}" if dcf_val else "N/A")
        if core.get("price") is not None and dcf_val:
            st.write(f"Market price: ${core.get('price'):.2f} → {'Undervalued ✅' if core.get('price') < dcf_val else 'Overvalued ❌'}")

        # Show DCF schedule breakdown
        def build_schedule(starting_fcf, g1, g2, tg, r, multiply=1.0):
            rows = []
            fcf = float(starting_fcf) * float(multiply)
            for y in range(1, 6):
                fcf = fcf * (1 + float(g1))
                pv = fcf / ((1 + r) ** y)
                rows.append({"Year": y, "FCF": fcf, "PV": pv})
            for y in range(6, 11):
                fcf = fcf * (1 + float(g2))
                pv = fcf / ((1 + r) ** y)
                rows.append({"Year": y, "FCF": fcf, "PV": pv})
            terminal_value = fcf * (1 + tg) / (r - tg) if r > tg else np.nan
            pv_terminal = terminal_value / ((1 + r) ** 10) if not np.isnan(terminal_value) else np.nan
            rows.append({"Year": "Terminal", "FCF": np.nan, "PV": pv_terminal, "TerminalValue": terminal_value})
            return pd.DataFrame(rows)

        schedule = build_schedule(starting_fcf, g1, g2, tg, discount, multiply)
        st.dataframe(schedule)

        # PE multiple fair value
        st.write("**PE multiple fair value**")
        pe_target = st.number_input("Target PE for simple PE valuation", value=float(ratios.get("P/E (info)") or 15.0), format="%.2f")
        eps_for_pe = eps_used
        pe_value = pe_fair_value(eps_for_pe, pe_target)
        st.metric("PE-based fair value (per share)", f"${pe_value:,.2f}" if pe_value else "N/A")

        # Competitor comparison (if provided)
        peers = [p.strip().upper() for p in competitor_input.split(",") if p.strip()]
        if peers:
            st.markdown("### Peer Comparison")
            peer_rows = []
            for p in peers:
                try:
                    pdict = fetch_yf(p)
                    pr = compute_common_ratios(pdict)
                    peer_rows.append({
                        "Ticker": p,
                        "Price": pr.get("Price"),
                        "P/E (info)": pr.get("P/E (info)"),
                        "EPS": pr.get("EPS (used)"),
                        "BVPS": pr.get("Book Value per Share (BVPS)"),
                        "MarketCap": pr.get("Market Cap (info)")
                    })
                except Exception:
                    peer_rows.append({"Ticker": p, "Price": "N/A", "P/E (info)": "N/A", "EPS": "N/A", "BVPS": "N/A", "MarketCap": "N/A"})
            peer_df = pd.DataFrame(peer_rows).set_index("Ticker")
            st.dataframe(peer_df)

        st.success("Analysis complete. Edit DCF inputs in the sidebar and re-run to test scenarios.")

else:
    st.info("Enter a ticker and click Run analysis. You can edit DCF inputs in the sidebar before running.")

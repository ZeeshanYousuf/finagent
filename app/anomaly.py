import os
import chromadb
import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="bank_statements")


def get_all_transactions(session_id: str = "default") -> list:
    """
    Fetch ALL transactions from ChromaDB for this session
    """
    collection_name = f"transactions_{session_id}"
    session_collection = chroma_client.get_or_create_collection(
        name=collection_name
    )
    count = session_collection.count()
    if count == 0:
        return []

    results = session_collection.get(
        limit=count,
        include=["documents", "metadatas"]
    )
    return results['documents']


def parse_transactions(documents: list) -> pd.DataFrame:
    """
    Parse raw text documents back into a DataFrame
    for numerical analysis
    """
    rows = []
    for doc in documents:
        try:
            # Parse "Date: X, Description: X, Amount: $X, Type: X, Source: X"
            parts = {}
            for part in doc.split(', '):
                if ': ' in part:
                    key, value = part.split(': ', 1)
                    parts[key.strip()] = value.strip()

            # Clean amount - remove $ sign
            amount_str = parts.get('Amount', '0').replace('$', '')
            amount = float(amount_str)

            rows.append({
                'date': parts.get('Date', ''),
                'description': parts.get('Description', ''),
                'amount': amount,
                'type': parts.get('Type', ''),
                'source': parts.get('Source', '')
            })
        except Exception:
            # Skip rows that can't be parsed
            continue

    return pd.DataFrame(rows)


def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Run anomaly detection rules on transactions.
    Returns a dictionary of detected anomalies.
    """
    anomalies = {}

    # Only look at debit transactions (spending)
    debits = df[df['amount'] < 0].copy()
    debits['amount_abs'] = debits['amount'].abs()

    if debits.empty:
        return anomalies

    # --- Rule 1: Large transactions (top 10% of spending) ---
    threshold = debits['amount_abs'].quantile(0.90)
    large_txns = debits[debits['amount_abs'] > threshold]
    if not large_txns.empty:
        anomalies['large_transactions'] = large_txns[[
            'date', 'description', 'amount_abs'
        ]].to_dict('records')

    # --- Rule 2: Duplicate charges (same amount + same description) ---
    duplicates = debits[
        debits.duplicated(subset=['description', 'amount'], keep=False)
    ]
    if not duplicates.empty:
        anomalies['duplicate_charges'] = duplicates[[
            'date', 'description', 'amount_abs'
        ]].to_dict('records')

    # --- Rule 3: Spending by category ---
    # Group by merchant/description
    category_spending = debits.groupby('description')['amount_abs'].sum()
    category_spending = category_spending.sort_values(ascending=False)
    anomalies['top_merchants'] = [
        {'merchant': merchant, 'total': round(total, 2)}
        for merchant, total in category_spending.head(5).items()
    ]

    # --- Rule 4: Total spending vs total income ---
    credits = df[df['amount'] > 0]
    total_income = credits['amount'].sum()
    total_spending = debits['amount_abs'].sum()

    if total_income > 0:
        spending_ratio = (total_spending / total_income) * 100
        anomalies['spending_ratio'] = {
            'total_income': round(total_income, 2),
            'total_spending': round(total_spending, 2),
            'percentage': round(spending_ratio, 1)
        }

    return anomalies


def format_anomalies_for_claude(anomalies: dict) -> str:
    """
    Format anomaly data into readable text for Claude
    """
    lines = []

    if 'large_transactions' in anomalies:
        lines.append("LARGE TRANSACTIONS (top 10% of spending):")
        for t in anomalies['large_transactions']:
            lines.append(f"  - {t['date']}: {t['description']} ${t['amount_abs']:.2f}")

    if 'duplicate_charges' in anomalies:
        lines.append("\nPOSSIBLE DUPLICATE CHARGES:")
        for t in anomalies['duplicate_charges']:
            lines.append(f"  - {t['date']}: {t['description']} ${t['amount_abs']:.2f}")

    if 'top_merchants' in anomalies:
        lines.append("\nTOP MERCHANTS BY SPENDING:")
        for m in anomalies['top_merchants']:
            lines.append(f"  - {m['merchant']}: ${m['total']:.2f}")

    if 'spending_ratio' in anomalies:
        r = anomalies['spending_ratio']
        lines.append(f"\nSPENDING SUMMARY:")
        lines.append(f"  - Total Income: ${r['total_income']:.2f}")
        lines.append(f"  - Total Spending: ${r['total_spending']:.2f}")
        lines.append(f"  - Spending as % of Income: {r['percentage']}%")

    return "\n".join(lines)


def generate_insights(session_id: str = "default") -> str:
    """
    Main function - detects anomalies and asks Claude
    to generate actionable financial insights
    """
    print("Fetching all transactions...")
    documents = get_all_transactions(session_id)

    if not documents:
        return "No transactions found in database."

    print(f"Analyzing {len(documents)} transactions...")
    df = parse_transactions(documents)

    print("Running anomaly detection...")
    anomalies = detect_anomalies(df)

    anomaly_text = format_anomalies_for_claude(anomalies)
    print("Anomalies detected:")
    print(anomaly_text)

    # Ask Claude to interpret the anomalies
    print("\nAsking Claude for insights...")
    prompt = f"""You are a personal finance advisor analyzing someone's bank transactions.
Based on the financial analysis below, provide 3-5 specific, actionable insights.
Be direct, specific, and helpful. Flag any concerns and suggest improvements.

FINANCIAL ANALYSIS:
{anomaly_text}

Provide insights in this format:
1. [Category] Insight title
   Details and recommendation

Keep each insight concise but specific with actual numbers."""

    message = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


# Test directly
if __name__ == "__main__":
    insights = generate_insights()
    print("\n" + "="*50)
    print("FINANCIAL INSIGHTS:")
    print("="*50)
    print(insights)
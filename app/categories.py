import os
import chromadb
import anthropic
import pandas as pd
from dotenv import load_dotenv
from app.anomaly import get_all_transactions, parse_transactions

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Category rules - keyword matching first (fast + free)
# Claude is used as fallback for unknown merchants
CATEGORY_RULES = {
    "Housing": [
        "pennymac", "mortgage", "rent", "hoa", "tierra montana",
        "apartment", "realty"
    ],
    "Food & Dining": [
        "burger king", "mcdonald", "wendy", "taco bell", "chipotle",
        "subway", "pizza", "restaurant", "dining", "doordash",
        "grubhub", "uber eats", "starbucks", "dunkin", "sonic",
        "chick-fil", "panera", "sams club", "costco", "walmart",
        "grocery", "safeway", "kroger", "fry's", "sprouts", "cvs pharmacy",
        "walgreens"
    ],
    "Transport": [
        "gas station", "shell", "chevron", "arco", "circle k",
        "uber", "lyft", "parking", "toll", "mvd", "auto"
    ],
    "Bills & Utilities": [
        "electric", "water", "gas bill", "internet", "verizon",
        "at&t", "t-mobile", "cox", "centurylink", "salt river",
        "city of phoenix", "aps", "srp"
    ],
    "Healthcare": [
        "hospital", "medical", "pharmacy", "cvs", "walgreen",
        "doctor", "dental", "vision", "health", "children's"
    ],
    "Shopping": [
        "amazon", "target", "best buy", "home depot", "lowes",
        "apple.com", "google", "microsoft", "online"
    ],
    "Transfers & Payments": [
        "wise", "zelle", "venmo", "paypal", "transfer",
        "bank of america", "chase", "wells fargo", "applecard",
        "credit card", "payment", "discover"
    ],
    "Income": [
        "payroll", "salary", "deposit", "direct dep"
    ]
}


def categorize_transaction(description: str) -> str:
    """
    Categorize a transaction using keyword matching.
    Returns category name.
    """
    description_lower = description.lower()

    for category, keywords in CATEGORY_RULES.items():
        for keyword in keywords:
            if keyword in description_lower:
                return category

    return "Other"


def categorize_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add category column to DataFrame
    """
    df['category'] = df['description'].apply(categorize_transaction)
    return df


def get_category_summary(df: pd.DataFrame) -> dict:
    """
    Group transactions by category and calculate totals
    """
    summary = {}

    for category in df['category'].unique():
        cat_df = df[df['category'] == category]

        # Separate income from spending
        spending = cat_df[cat_df['amount'] < 0]['amount'].sum()
        income = cat_df[cat_df['amount'] > 0]['amount'].sum()

        transactions = []
        for _, row in cat_df.iterrows():
            transactions.append({
                'date': row['date'],
                'description': row['description'],
                'amount': round(row['amount'], 2)
            })

        summary[category] = {
            'total_spending': round(abs(spending), 2),
            'total_income': round(income, 2),
            'transaction_count': len(cat_df),
            'transactions': transactions
        }

    return summary


def generate_category_report(session_id: str = "default") -> str:
    """
    Generate a full spending report by category
    """
    print("Fetching transactions...")
    documents = get_all_transactions(session_id)
    df = parse_transactions(documents)

    print("Categorizing transactions...")
    df = categorize_all(df)

    print("Building category summary...")
    summary = get_category_summary(df)

    # Format summary for Claude
    summary_text = []
    for category, data in summary.items():
        if data['total_spending'] > 0 or data['total_income'] > 0:
            summary_text.append(
                f"{category}:"
                f" spent=${data['total_spending']:.2f},"
                f" income=${data['total_income']:.2f},"
                f" transactions={data['transaction_count']}"
            )

    summary_str = "\n".join(summary_text)
    print("Category summary:")
    print(summary_str)

    # Ask Claude to write a clear report
    prompt = f"""You are a personal finance advisor.
Based on this spending breakdown by category, write a clear monthly spending report.

SPENDING BY CATEGORY:
{summary_str}

Write a concise report that:
1. Shows spending by category with amounts
2. Identifies the top 3 spending categories
3. Gives one specific recommendation per top category
4. Ends with a overall financial health summary

Use clear headings and be specific with numbers."""

    print("\nAsking Claude to write report...")
    message = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


if __name__ == "__main__":
    report = generate_category_report()
    print("\n" + "="*50)
    print("SPENDING REPORT:")
    print("="*50)
    print(report)
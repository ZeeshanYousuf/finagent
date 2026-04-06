import os
import pandas as pd
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="bank_statements")


def load_bank_csv(file_path: str) -> pd.DataFrame:
    """
    Load current account statement CSV.
    Skips summary rows at the top and normalizes column names.
    """
    # Read raw file first to find where real data starts
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the line number where "Date" header row is
    header_row = None
    for i, line in enumerate(lines):
        if line.startswith('Date'):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find 'Date' header row in bank CSV")

    #header_row = header_row+1
    print(f"Found header at line {header_row}, skipping {header_row} summary rows")

    # Now read CSV starting from the actual header row
    df = pd.read_csv(file_path, skiprows=range(header_row))
    df.columns = df.columns.str.strip()
    print(df.columns)
    # Keep only rows where Date looks like MM/DD/YYYY
    df = df[df['Date'].str.match(r'\d{2}/\d{2}/\d{4}', na=False)]

#     print(df)

    # Normalize to standard format
    normalized = pd.DataFrame({
        'date': df['Date'],
        'description': df['Description'],
        'amount': pd.to_numeric(
            df['Amount'].astype(str).str.replace(',', ''),
            errors='coerce'
        ),
        'type': pd.to_numeric(
            df['Amount'].astype(str).str.replace(',', ''),
            errors='coerce'
        ).apply(lambda x: 'credit' if x > 0 else 'debit'),
        'source': 'bank'
    })

    print(f"Loaded {len(normalized)} bank transactions from {file_path}")
    return normalized


def load_credit_csv(file_path: str) -> pd.DataFrame:
    """
    Load credit card statement CSV.
    Normalizes column names to standard format.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Normalize to standard format
    normalized = pd.DataFrame({
        'date': df['Posted Date'],
        'description': df['Payee'],
        'amount': pd.to_numeric(df['Amount'].astype(str).str.replace(',', ''), errors='coerce'),
        'type': pd.to_numeric(
            df['Amount'].astype(str).str.replace(',', ''),
            errors='coerce'
        ).apply(lambda x: 'credit' if x > 0 else 'debit'),
        'source': 'credit_card'
    })

    print(f"Loaded {len(normalized)} credit card transactions from {file_path}")
    return normalized


def dataframe_to_documents(df: pd.DataFrame) -> list:
    """
    Convert each transaction row into a readable text document
    for ChromaDB to store and search
    """
    documents = []

    for index, row in df.iterrows():
        text = (
            f"Date: {row['date']}, "
            f"Description: {row['description']}, "
            f"Amount: ${row['amount']}, "
            f"Type: {row['type']}, "
            f"Source: {row['source']}"
        )
        documents.append(text)

    return documents


def ingest_all(bank_file: str = "app/data/sample_bank.csv",
               credit_file: str = "app/data/sample_credit.csv",
               session_id: str = "default") -> int:
    """
    Main function - loads both CSVs, combines them,
    and stores everything in ChromaDB
    Supports per-session collections for multi-user
    """
    # Get or create collection for this session
    collection_name = f"transactions_{session_id}"
    session_collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    # Load both files
    bank_df = load_bank_csv(bank_file)
    credit_df = load_credit_csv(credit_file)

    # Combine both into one DataFrame
    combined_df = pd.concat([bank_df, credit_df], ignore_index=True)
    print(f"\nTotal transactions combined: {len(combined_df)}")

    # Convert to text documents
    documents = dataframe_to_documents(combined_df)

    # Clear existing data for this session
    existing = session_collection.count()
    if existing > 0:
        print(f"Clearing {existing} existing documents...")
        ids_to_delete = [f"txn_{i}" for i in range(existing)]
        session_collection.delete(ids=ids_to_delete)

    # Store in ChromaDB
    session_collection.add(
        documents=documents,
        ids=[f"txn_{i}" for i in range(len(documents))],
        metadatas=[{"source": row['source']} for _, row in combined_df.iterrows()]
    )

    print(f"Successfully stored {len(documents)} transactions in ChromaDB collection: {collection_name}")
    return len(documents)


if __name__ == "__main__":
    ingest_all()
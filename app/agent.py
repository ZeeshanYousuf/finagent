import os
import chromadb
import anthropic
from dotenv import load_dotenv
from app.logger import logger

load_dotenv()

# Initialize clients
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    logger.info("ChromaDB connected successfully")
except Exception as e:
    logger.error(f"Failed to connect to ChromaDB: {e}")
    raise

try:
    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    logger.info("Anthropic client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {e}")
    raise

# Conversation memory store
conversations = {}

def get_collection(session_id: str):
    """Get ChromaDB collection for this session"""
    collection_name = f"transactions_{session_id}"
    return chroma_client.get_or_create_collection(name=collection_name)

def search_transactions(query: str, session_id: str = "default", n_results: int = 10) -> list:
    """Search ChromaDB for relevant transactions"""
    try:
        collection = get_collection(session_id)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        logger.info(f"Search query: '{query}' returned {len(results['documents'][0])} results")
        return results['documents'][0]
    except Exception as e:
        logger.error(f"ChromaDB search failed: {e}")
        raise ValueError(f"Failed to search transactions: {str(e)}")


def get_or_create_session(session_id: str) -> list:
    """
    Get existing conversation or create new one.
    Returns the message history for this session.
    """
    if session_id not in conversations:
        conversations[session_id] = []
        logger.info(f"New session created: {session_id}")
    return conversations[session_id]


def clear_session(session_id: str):
    """Clear conversation history AND ChromaDB data for a session"""
    # Clear conversation memory
    if session_id in conversations:
        del conversations[session_id]
        logger.info(f"Conversation cleared: {session_id}")

    # Clear ChromaDB collection
    try:
        collection_name = f"transactions_{session_id}"
        chroma_client.delete_collection(collection_name)
        logger.info(f"ChromaDB collection deleted: {collection_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection {session_id}: {e}")


def answer_question(question: str, session_id: str = "default") -> str:
    """
    Main RAG function with memory.
    session_id allows multiple independent conversations.
    """
    logger.info(f"Session: {session_id} | Question: {question}")


    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if len(question) > 1000:
        raise ValueError("Question is too long (max 1000 characters)")

    try:
        # Get conversation history
        # Step 1: Get or create conversation history
        history = get_or_create_session(session_id)

        # Limit history size to prevent token explosion
        # Keep last 10 messages (5 exchanges)
        max_history = 10
        if len(history) > max_history:
            history = history[-max_history:]
            conversations[session_id] = history
            logger.info(f"History trimmed to {max_history} messages for session: {session_id}")

        # Search transactions
        # Step 2: Search for relevant transactions
        transactions = search_transactions(question, session_id=session_id)

        if not transactions:
            logger.warning(f"No transactions found for query: {question}")
            return "I couldn't find any relevant transactions for your question."


        # Step 3: Format transactions for Claude
        transactions_text = "\n".join(transactions)

        # Step 4: Build system prompt with financial data
        # System prompt = Claude's instructions and context
        system_prompt = f"""You are a helpful personal finance assistant.
You have access to the user's bank transactions below.
Use ONLY this data to answer questions.
If the answer cannot be found in the transactions, say so clearly.
Remember previous questions in our conversation and refer to them when relevant.

AVAILABLE TRANSACTIONS:
{transactions_text}"""

        # Step 5: Add current question to history
        history.append({
            "role": "user",
            "content": question
        })

        # Step 6: Send full conversation history to Claude
        # This is what gives Claude memory - we send ALL previous messages
        logger.info(f"Calling Claude API for session: {session_id}")
        message = claude.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            system=system_prompt,
            messages=history  # ← sending full history not just current question
        )

        # Step 7: Get Claude's response
        answer = message.content[0].text

        # Step 8: Add Claude's response to history
        # So next question Claude remembers this answer too
        history.append({
            "role": "assistant",
            "content": answer
        })

        logger.info(f"Answer generated successfully | Session: {session_id} | "
                f"Tokens used: {message.usage.input_tokens + message.usage.output_tokens}")
        return answer

    except ValueError:
        raise
    except anthropic.APIConnectionError as e:
        logger.error(f"Anthropic API connection error: {e}")
        raise ValueError("Could not connect to AI service. Please try again.")
    except anthropic.RateLimitError as e:
        logger.error(f"Anthropic rate limit hit: {e}")
        raise ValueError("AI service is busy. Please wait a moment and try again.")
    except anthropic.APIStatusError as e:
        logger.error(f"Anthropic API error {e.status_code}: {e.message}")
        raise ValueError(f"AI service error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error in answer_question: {e}", exc_info=True)
        raise ValueError("An unexpected error occurred. Please try again.")


if __name__ == "__main__":
    # Test conversation with memory
    session = "test_session"

    questions = [
        "How much did I spend on food?",
        "Which of those was the most expensive single transaction?",
        "What percentage of my income did that represent?",
    ]

    for question in questions:
        answer = answer_question(question, session_id=session)
        print(f"\nAnswer: {answer}")
        print("-" * 50)
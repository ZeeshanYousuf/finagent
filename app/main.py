from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import time
from app.agent import answer_question, clear_session
from app.ingest import ingest_all, load_bank_csv, load_credit_csv, dataframe_to_documents
from app.anomaly import generate_insights
from app.categories import generate_category_report
from app.logger import logger
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create FastAPI app - like creating a new Laravel/CakePHP application
app = FastAPI(
    title="FinAgent API",
    description="AI-powered personal finance assistant",
    version="1.0.0"
)

# Serve static files - like Apache serving public/ folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Allow all origins for development
# Like CORS settings in your PHP apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model - defines what the request body should look like
# Like a PHP DTO (Data Transfer Object) or Form Request
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class QuestionResponse(BaseModel):
    question: str
    answer: str
    status: str = "success"


# Routes - like controllers in CakePHP/Laravel
@app.get("/")
def root():
    logger.info("Root accessed - redirecting to chat")
    return FileResponse("app/static/index.html")

@app.get("/chat")
def chat_page():
    """Serve the chat UI"""
    return FileResponse("app/static/index.html")


@app.get("/transactions/count")
def get_transaction_count(session_id: str = "default"):
    """Returns how many transactions are in the database"""
    try:
        collection_name = f"transactions_{session_id}"
        session_collection = chroma_client.get_or_create_collection(
            name=collection_name
        )
        count = session_collection.count()
        logger.info(f"Transaction count requested: {count} for session: {session_id}")
        return {"total_transactions": count}
    except Exception as e:
        logger.error(f"Failed to get transaction count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get transaction count")


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """
    Main endpoint - ask a question about your finances
    POST /ask
    Body: {"question": "How much did I spend on food?"}
    """
    logger.info(f"Question received | Session: {request.session_id} | Q: {request.question}")
    start_time = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = answer_question(request.question, session_id=request.session_id)
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Question answered in {elapsed}s | Session: {request.session_id}")
        return QuestionResponse(
            question=request.question,
            answer=answer
        )
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.get("/insights")
def get_insights(session_id: str = "default"):
    logger.info(f"Insights requested for session: {session_id}")
    start_time = time.time()
    try:
        insights = generate_insights(session_id)
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Insights generated in {elapsed}s")
        return {"status": "success", "insights": insights}
    except Exception as e:
        logger.error(f"Insights generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@app.post("/ingest/csv")
def ingest_csv_file(
    file: UploadFile = File(...),
    session_id: str = "default",
    file_type: str = "bank"
):
    """
    Upload and ingest a CSV file for a specific session
    file_type: 'bank' or 'credit'
    """
    logger.info(f"CSV upload received: {file.filename} for session: {session_id}")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        # Save uploaded file temporarily
        temp_path = f"app/data/upload_{session_id}_{file_type}.csv"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Determine which loader to use
        if file_type == "credit":
            df = load_credit_csv(temp_path)
        else:
            df = load_bank_csv(temp_path)

        # Get session collection
        collection_name = f"transactions_{session_id}"
        session_collection = chroma_client.get_or_create_collection(
            name=collection_name
        )

        # Convert to documents
        documents = dataframe_to_documents(df)

        # Add to existing collection (don't clear — user might upload both files)
        existing = session_collection.count()
        start_id = existing

        session_collection.add(
            documents=documents,
            ids=[f"txn_{start_id + i}" for i in range(len(documents))],
            metadatas=[{"source": file_type} for _ in documents]
        )

        # Clean up temp file
        os.remove(temp_path)

        logger.info(f"CSV ingested: {len(documents)} transactions for session: {session_id}")
        return {
            "status": "success",
            "message": f"Successfully ingested {len(documents)} transactions",
            "filename": file.filename,
            "total": existing + len(documents)
        }
    except Exception as e:
        logger.error(f"CSV ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest CSV: {str(e)}")


@app.post("/ingest/reload")
def reload_all(session_id: str = "default"):
    logger.info(f"Reload requested for session: {session_id}")
    try:
        count = ingest_all(session_id=session_id)
        logger.info(f"Reload complete: {count} transactions")
        return {"status": "success", "message": f"Successfully reloaded {count} transactions"}
    except Exception as e:
        logger.error(f"Reload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload: {str(e)}")

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clear all data for a session — conversation + ChromaDB"""
    try:
        clear_session(session_id)
        logger.info(f"Session fully deleted: {session_id}")
        return {"status": "success", "message": f"Session {session_id} cleared"}
    except Exception as e:
        logger.error(f"Failed to clear session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session")

@app.get("/report")
def get_report(session_id: str = "default"):
    logger.info(f"Report requested for session: {session_id}")
    start_time = time.time()
    try:
        report = generate_category_report(session_id)
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Report generated in {elapsed}s")
        return {"status": "success", "report": report}
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@app.post("/session/init")
def init_session(session_id: str = "default"):
    """
    Initialize a new session with sample data
    Called when a new user opens the chat
    """
    logger.info(f"Initializing session: {session_id}")
    try:
        # Check if session already has data
        collection_name = f"transactions_{session_id}"
        session_collection = chroma_client.get_or_create_collection(
            name=collection_name
        )
        count = session_collection.count()

        if count > 0:
            logger.info(f"Session {session_id} already has {count} transactions")
            return {"status": "success", "transactions": count, "loaded": "existing"}

        # Load sample data for new session
        count = ingest_all(session_id=session_id)
        logger.info(f"Session {session_id} initialized with {count} sample transactions")
        return {"status": "success", "transactions": count, "loaded": "sample"}

    except Exception as e:
        logger.error(f"Session init failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {str(e)}")
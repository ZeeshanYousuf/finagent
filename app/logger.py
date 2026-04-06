import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Log filename with today's date
log_filename = f"logs/finagent_{datetime.now().strftime('%Y-%m-%d')}.log"

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        # Write to file
        logging.FileHandler(log_filename),
        # Also print to terminal
        logging.StreamHandler()
    ]
)

# Create named logger for our app
logger = logging.getLogger("finagent")
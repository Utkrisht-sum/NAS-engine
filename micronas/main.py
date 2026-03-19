import os
import sys

# Ensure python path includes the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from gui.app import run_app
from utils.logger import get_logger

logger = get_logger("Main")

if __name__ == "__main__":
    logger.info("Initializing MICRONAS Engine GUI...")
    run_app()

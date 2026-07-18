import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.Orchestrator.ErrorHandler')


class ErrorHandler:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.error_count = 0
        self.MAX_ERRORS = 5

    async def handle_error(self, error: Exception, context: str):
        self.error_count += 1
        logger.error(f"Error in {context}: {error} (Total Errors: {self.error_count})")
        if self.error_count >= self.MAX_ERRORS:
            await self.shutdown_system(f"Critical error threshold reached: {error}")
        else:
            await self.attempt_recovery(error, context)

    async def attempt_recovery(self, error: Exception, context: str):
        logger.info(f"Attempting recovery for error in {context}...")
        if isinstance(error, ConnectionError):
            logger.info("Recovering from connection error.")
        elif isinstance(error, asyncio.TimeoutError):
            logger.info("Recovering from timeout.")
        else:
            logger.info("Generic recovery path.")

    async def shutdown_system(self, reason: str):
        logger.critical(f"SYSTEM SHUTDOWN INITIATED: {reason}")
        if self.orchestrator:
            await self.orchestrator.stop()
        raise SystemExit(f"Fatal Error: {reason}")

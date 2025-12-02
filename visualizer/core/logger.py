"""
Pipeline Logging System.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class PipelineLogger:
    """Pipeline-specific logger."""

    def __init__(self, name: str, log_file: Optional[Path] = None, verbose: bool = True):
        """
        Args:
            name: Logger name
            log_file: Optional log file path
            verbose: Whether to log to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console output
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File output
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Info log."""
        self.logger.info(message)

    def debug(self, message: str):
        """Debug log."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Warning log."""
        self.logger.warning(message)

    def error(self, message: str):
        """Error log."""
        self.logger.error(message)

    def success(self, message: str):
        """Success log (INFO level)."""
        self.logger.info(f"✅ {message}")

    def stage(self, stage_name: str, stage_num: Optional[int] = None):
        """Stage log."""
        separator = "=" * 70
        if stage_num is not None:
            self.logger.info(f"\n{separator}")
            self.logger.info(f"Stage {stage_num}: {stage_name}")
            self.logger.info(separator)
        else:
            self.logger.info(f"\n{separator}")
            self.logger.info(stage_name)
            self.logger.info(separator)

    def phase(self, phase_name: str, phase_num: Optional[int] = None):
        """Sub-task log within a stage."""
        if phase_num is not None:
            self.logger.info(f"\n--- Phase {phase_num}: {phase_name} ---")
        else:
            self.logger.info(f"\n--- {phase_name} ---")

    def progress(self, current: int, total: int, item_name: str = "item"):
        """Progress log."""
        percentage = (current / total) * 100
        self.logger.info(f"[{current}/{total}] ({percentage:.0f}%) {item_name}")


def setup_logger(
    name: str = "pipeline",
    log_file: Optional[Path] = None,
    verbose: bool = True
) -> PipelineLogger:
    """
    Quickly configure a PipelineLogger.
    """
    return PipelineLogger(name, log_file, verbose)


# Preset log styles
class LogStyles:
    """Predefined log message styles."""

    @staticmethod
    def header(text: str) -> str:
        """Header."""
        return f"\n{'='*70}\n{text}\n{'='*70}"

    @staticmethod
    def subheader(text: str) -> str:
        """Subheader."""
        return f"\n{'-'*50}\n{text}\n{'-'*50}"

    @staticmethod
    def success(text: str) -> str:
        """Success message."""
        return f"✅ {text}"

    @staticmethod
    def error(text: str) -> str:
        """Error message."""
        return f"❌ {text}"

    @staticmethod
    def warning(text: str) -> str:
        """Warning message."""
        return f"⚠️  {text}"

    @staticmethod
    def info(text: str) -> str:
        """Info message."""
        return f"ℹ️  {text}"

    @staticmethod
    def progress(current: int, total: int, desc: str = "") -> str:
        """Progress message."""
        percentage = (current / total) * 100
        bar_length = 30
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        return f"[{bar}] {percentage:.0f}% ({current}/{total}) {desc}"

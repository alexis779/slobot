"""Work around LeRobot logging that drops exception tracebacks."""

from __future__ import annotations

import logging

from lerobot.utils import utils as lerobot_utils

_PATCHED = False


def _patch_handlers_for_exc_info() -> None:
    for handler in logging.getLogger().handlers:
        formatter = handler.formatter
        if formatter is None or getattr(formatter, "_slobot_exc_patched", False):
            continue

        base_format = formatter.format

        def format_record(record: logging.LogRecord, _base=base_format) -> str:
            text = _base(record)
            if record.exc_info:
                text += "\n" + logging.Formatter().formatException(record.exc_info)
            elif record.exc_text:
                text += "\n" + record.exc_text
            return text

        formatter.format = format_record
        formatter._slobot_exc_patched = True


def patch_init_logging_for_tracebacks(lerobot_actor_module=None) -> None:
    """LeRobot ``init_logging`` uses a custom formatter that omits ``exc_info``."""
    global _PATCHED
    if _PATCHED:
        return

    orig_init_logging = lerobot_utils.init_logging

    def init_logging(*args, **kwargs):
        orig_init_logging(*args, **kwargs)
        _patch_handlers_for_exc_info()

    lerobot_utils.init_logging = init_logging
    if lerobot_actor_module is not None:
        lerobot_actor_module.init_logging = init_logging
    _PATCHED = True

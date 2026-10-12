import logging

from xps_afruz.logging_config import configure_application_logging


def test_application_log_is_bounded_and_written_to_requested_directory(tmp_path) -> None:
    logger, log_path = configure_application_logging(tmp_path, level=logging.DEBUG)

    logger.warning("diagnostic smoke message")
    for handler in logger.handlers:
        handler.flush()

    assert log_path.parent == tmp_path
    assert "diagnostic smoke message" in log_path.read_text(encoding="utf-8")
    rotating = next(handler for handler in logger.handlers if hasattr(handler, "maxBytes"))
    assert rotating.maxBytes == 2 * 1024 * 1024
    assert rotating.backupCount == 4

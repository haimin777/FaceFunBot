#!/usr/bin/env python3
"""Telegram bot wrapper for FaceFunBot."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from telegram import PhotoSize, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from couple_match import MatchMode, compare_photos, format_report


TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
UPLOAD_DIR_ENV = "FACEFUNBOT_UPLOAD_DIR"
DEFAULT_UPLOAD_DIR = Path("uploaded_photos")
DEFAULT_MODE: MatchMode = "face"


def current_mode(context: ContextTypes.DEFAULT_TYPE) -> MatchMode:
    return context.chat_data.get("mode", DEFAULT_MODE)


def photo_queue(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    return context.chat_data.setdefault("photos", [])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["mode"] = DEFAULT_MODE
    context.chat_data["photos"] = []
    await update.effective_message.reply_text(
        "Send me 2 photos and I will calculate your completely unserious couple match.\n\n"
        "Commands:\n"
        "/face - compare face photos\n"
        "/palm - compare palm photos\n"
        "/reset - clear the current pair"
    )


async def set_face_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await set_mode(update, context, "face")


async def set_palm_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await set_mode(update, context, "palm")


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE, mode: MatchMode) -> None:
    context.chat_data["mode"] = mode
    context.chat_data["photos"] = []
    label = "faces" if mode == "face" else "palms"
    await update.effective_message.reply_text(f"Mode set to {mode}. Send me 2 {label}.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cleanup_photos(photo_queue(context))
    context.chat_data["photos"] = []
    await update.effective_message.reply_text("Pair cleared. Send me 2 fresh photos.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.photo:
        return

    queue = photo_queue(context)
    mode = current_mode(context)
    photo = message.photo[-1]
    telegram_file = await photo.get_file()

    suffix = Path(telegram_file.file_path or "").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(prefix="facefunbot_", suffix=suffix, delete=False) as temp:
        temp_path = Path(temp.name)

    await telegram_file.download_to_drive(custom_path=temp_path)
    archive_photo(temp_path, mode, update, photo, suffix)
    queue.append(str(temp_path))

    if len(queue) == 1:
        label = "face" if mode == "face" else "palm"
        await message.reply_text(f"Got the first {label}. Send the second one.")
        return

    first, second = Path(queue[0]), Path(queue[1])
    context.chat_data["photos"] = []
    await message.reply_text("Comparing the cosmic evidence...")

    try:
        report = await asyncio.to_thread(compare_photos, first, second, mode)
        await message.reply_text(format_report(report))
    except Exception:
        logging.exception("Photo comparison failed")
        await message.reply_text("I could not compare those photos. Try two clearer images.")
    finally:
        cleanup_photos([str(first), str(second)])


async def handle_non_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = current_mode(context)
    queued = len(photo_queue(context))
    needed = 2 - queued
    label = "face photos" if mode == "face" else "palm photos"
    await update.effective_message.reply_text(f"Send {needed} more {label} to get a match.")


def cleanup_photos(paths: list[str]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logging.warning("Could not remove temporary photo %s", path)


def archive_photo(source: Path, mode: MatchMode, update: Update, photo: PhotoSize, suffix: str) -> Path:
    folder_name = "faces" if mode == "face" else "palms"
    archive_dir = Path(os.environ.get(UPLOAD_DIR_ENV, DEFAULT_UPLOAD_DIR)) / folder_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    chat_id = update.effective_chat.id if update.effective_chat else "unknown_chat"
    user_id = update.effective_user.id if update.effective_user else "unknown_user"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_unique_id = safe_filename_part(photo.file_unique_id)
    destination = archive_dir / f"{timestamp}_chat-{chat_id}_user-{user_id}_{safe_unique_id}{suffix}"
    shutil.copy2(source, destination)
    logging.info("Archived %s photo to %s", mode, destination)
    return destination


def safe_filename_part(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
    return safe[:80] or "photo"


def build_application(token: str) -> Application:
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("face", set_face_mode))
    application.add_handler(CommandHandler("palm", set_palm_mode))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_non_photo))
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"Set {TOKEN_ENV} before running the bot.")

    build_application(token).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

"""
Telegram Bot for horse betting predictions and race results.
"""
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from rich.console import Console
from rich.table import Table

from ..data.pipeline import DataPipeline
from ..ml.models import HorseBetModel

log = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Telegram Bot handlers
# ---------------------------------------------------------------------------

class HorseBetBot:
    """Telegram bot for horse racing predictions."""

    def __init__(self, token: str, db_path: str, model_path: str = None):
        self.token = token
        self.db_path = db_path
        self.model_path = model_path
        self.pipeline = DataPipeline(db_path)
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = HorseBetModel()
            try:
                self.model.load(model_path)
                log.info("Model loaded successfully")
            except Exception as e:
                log.warning(f"Could not load model: {e}")

    def build_app(self):
        """Build the Telegram application."""
        app = Application.builder().token(self.token).build()

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("race", self.cmd_race))
        app.add_handler(CommandHandler("predict", self.cmd_predict))
        app.add_handler(CommandHandler("results", self.cmd_results))
        app.add_handler(CommandHandler("schedule", self.cmd_schedule))
        app.add_handler(CommandHandler("reload", self.cmd_reload))
        app.add_handler(CallbackQueryHandler(self.on_callback))

        return app

    # ---- Commands ----

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🐴 *Horse Bet Bot*\n\n"
            "Welcome! I predict horse racing outcomes and show results.\n\n"
            "*Commands:*\n"
            "/race YYYY/MM/DD ST — Get race card for a day\n"
            "/predict YYYY/MM/DD ST — Get ML predictions\n"
            "/results YYYY/MM/DD ST — Get race results\n"
            "/schedule — Upcoming races\n"
            "/reload — Reload ML model\n"
            "/help — Show all commands",
            parse_mode='Markdown'
        )

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🐴 *Horse Bet Commands*\n\n"
            "`/race <date> <venue>` — Race card (e.g. /race 2026/04/19 ST)\n"
            "`/predict <date> <venue>` — ML predictions for race\n"
            "`/results <date> <venue>` — Official results + dividends\n"
            "`/schedule` — Next race day schedule\n"
            "`/reload` — Reload prediction model\n\n"
            "*Venues:* `ST` = Sha Tin, `HV` = Happy Valley\n"
            "*Date format:* YYYY/MM/DD",
            parse_mode='Markdown'
        )

    async def cmd_race(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show race card for a specific day."""
        args = ctx.args
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: `/race 2026/04/19 ST`\n"
                "Example: `/race 2026/04/19 ST`",
                parse_mode='Markdown'
            )
            return

        race_date = args[0]
        venue = args[1].upper()

        try:
            data = self.pipeline.scraper.get_race_day_results(race_date, venue)
            races = data.get('races', [])

            if not races:
                await update.message.reply_text(f"No races found for {race_date} {venue}")
                return

            msg = f"📋 *Race Card — {race_date} {venue}*\n"
            msg += f"Total races: {data.get('total_races', len(races))}\n\n"

            for r in races[:10]:
                msg += f"Race {r['race_no']}: {r['text']}\n"

            await update.message.reply_text(msg, parse_mode='Markdown')

        except Exception as e:
            log.error(f"Race card error: {e}")
            await update.message.reply_text(f"Error fetching race card: {e}")

    async def cmd_predict(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show ML predictions for a race day."""
        if not self.model:
            await update.message.reply_text(
                "⚠️ Model not loaded. Use /reload or train first with `python -m ml.train`"
            )
            return

        args = ctx.args
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: `/predict 2026/04/19 ST`\n"
                "Shows top picks for each race based on ML model.",
                parse_mode='Markdown'
            )
            return

        race_date = args[0]
        venue = args[1].upper()

        try:
            # Load recent data
            df = self.pipeline.get_recent_races(days=90)

            if df.empty:
                await update.message.reply_text("No historical data available. Collect data first.")
                return

            # Filter + predict
            df_race = df[(df['race_date'].astype(str) == race_date.replace('/', '-')) &
                         (df['venue'] == venue)]

            if df_race.empty:
                await update.message.reply_text(
                    f"No data for {race_date} {venue}. "
                    "Try scraping first or check date format."
                )
                return

            preds = self.model.predict(df_race)

            msg = f"🎯 *ML Predictions — {race_date} {venue}*\n\n"

            for (date, ven, race_no), group in preds.groupby(
                ['race_date', 'venue', 'race_no']
            ):
                msg += f"*Race {race_no}*\n"
                top = group.head(4)
                for _, row in top.iterrows():
                    odds = f"{row['win_odds']:.1f}" if pd.notna(row['win_odds']) else "N/A"
                    prob = f"{row['prob_win_norm'] * 100:.1f}%"
                    emoji = "🥇" if _ == 0 else "🥈" if _ == 1 else "🥉" if _ == 2 else "  "
                    msg += f"{emoji} {row['horse_name']} | ODDS {odds} | P(WIN) {prob}\n"
                msg += "\n"

            await update.message.reply_text(msg, parse_mode='Markdown')

        except Exception as e:
            log.error(f"Predict error: {e}")
            await update.message.reply_text(f"Error generating predictions: {e}")

    async def cmd_results(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show official race results."""
        args = ctx.args
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: `/results 2026/04/19 ST`",
                parse_mode='Markdown'
            )
            return

        race_date = args[0]
        venue = args[1].upper()

        try:
            # Load from DB
            df = self.pipeline.get_recent_races(days=30)
            df_race = df[(df['race_date'].astype(str) == race_date.replace('/', '-')) &
                         (df['venue'] == venue)]

            if df_race.empty:
                await update.message.reply_text(
                    f"No results found for {race_date} {venue}. "
                    "Try: `/race {date} {venue}` to get the race card first."
                )
                return

            msg = f"🏁 *Results — {race_date} {venue}*\n\n"

            for (date, ven, race_no), group in df_race.groupby(
                ['race_date', 'venue', 'race_no']
            ):
                finished = group.dropna(subset=['finishing_position']).sort_values('finishing_position')
                if finished.empty:
                    continue

                msg += f"*Race {race_no}*\n"
                for _, row in finished.head(3).iterrows():
                    pos_emoji = "🥇" if row['finishing_position'] == 1 else \
                                "🥈" if row['finishing_position'] == 2 else "🥉"
                    margin = row.get('margin', '') or ''
                    msg += f"{pos_emoji} {row['horse_name']} ({margin})\n"
                msg += "\n"

            await update.message.reply_text(msg, parse_mode='Markdown')

        except Exception as e:
            log.error(f"Results error: {e}")
            await update.message.reply_text(f"Error fetching results: {e}")

    async def cmd_schedule(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show upcoming race schedule."""
        try:
            schedule = self.pipeline.scraper.get_upcoming_races()
            if not schedule:
                await update.message.reply_text("No upcoming races found.")
                return

            msg = "📅 *Upcoming Race Days*\n\n"
            for entry in schedule[:10]:
                msg += f"{entry['date']} — {entry['venue']} ({entry['text']})\n"

            await update.message.reply_text(msg, parse_mode='Markdown')

        except Exception as e:
            log.error(f"Schedule error: {e}")
            await update.message.reply_text(f"Error fetching schedule: {e}")

    async def cmd_reload(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Reload ML model."""
        if not self.model_path:
            await update.message.reply_text("No model path configured.")
            return

        try:
            self.model = HorseBetModel()
            self.model.load(self.model_path)
            await update.message.reply_text("✅ Model reloaded successfully!")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to reload model: {e}")

    async def on_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()
        # Handle callback data
        await query.edit_message_text(text=f"Received: {query.data}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run the Telegram bot."""
    from dotenv import load_dotenv
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        console.print("[red]TELEGRAM_BOT_TOKEN not set in .env[/red]")
        return

    db_path = os.getenv("HORSE_DB_PATH", "data/horse.db")
    model_path = os.getenv("MODEL_PATH", "models/model.json")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    bot = HorseBetBot(token, db_path, model_path)
    app = bot.build_app()

    console.print("[green]🐴 Horse Bet Bot starting...[/green]")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()

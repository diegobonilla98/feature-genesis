from telegram_training_helper import TelegramBot


MESSAGE = "Features wake in lines\nSilent sketches shape their paths\nMeaning blooms at last"


bot = TelegramBot(auto_start=False)
bot.send_message(MESSAGE)

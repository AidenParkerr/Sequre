import os
import telebot

" A simple telegram bot that sends a message to a user."

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
  bot.reply_to(message, "Welcome to the bot!")


@bot.message_handler(commands=['help'])
def send_help(message):
  bot.reply_to(message, "Send a message to the bot to receive a response.")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
  bot.reply_to(message, message.text)


def main():
  bot.polling()


if __name__ == "__main__":
  main()

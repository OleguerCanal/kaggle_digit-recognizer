import yaml
import requests
# import file

class TelegramBot():
    def __init__(self, param_file="/home/oleguer/.telegram_bots/oleguer-hab.yaml"):
        stream = open(param_file, 'r')
        params = yaml.load(stream, Loader=yaml.FullLoader)
        self.bot_token = str(params["bot_token"])
        self.bot_chatID = str(params["bot_chatID"])

    def send(self, bot_message):
        try:
            send_text = 'https://api.telegram.org/bot' + self.bot_token + '/sendMessage?chat_id=' + self.bot_chatID + '&parse_mode=Markdown&text=' + bot_message
            response = requests.get(send_text)
        except Exception as e:
            print(e)
            return None
        return response.json()
        

if __name__ == "__main__":
    telegram_bot = TelegramBot()
    answer = telegram_bot.send("ke ase")
    print(answer)
from keras.callbacks import Callback
from telegram_bot.telegram_bot import TelegramBot

# Inherit callback class and overwrite on_epoch_end method
class TelegramSummary(Callback):
    def on_train_begin(self, logs={}):
        telegram_bot = TelegramBot()  #TODO(oleguer): Initialize it before
        answer = telegram_bot.send("Starting training...")
        return

    def on_epoch_end(self, epoch, logs={}):
        print("Sending results to telegram")
        loss = "loss: " + str(round(logs.get('loss'), 2))
        acc = "acc:  " + str(round(logs.get('acc'), 2))
        val_loss = "val loss: " + str(round(logs.get('val_loss'), 2))
        val_acc = "val acc:  " + str(round(logs.get('val_acc'), 2))

        summary = loss + "  " + val_loss + "\n" + acc + "  " + val_acc
        
        telegram_bot = TelegramBot()  #TODO(oleguer): Initialize it before
        answer = telegram_bot.send(summary)
        return


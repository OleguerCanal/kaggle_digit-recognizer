from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from telegram_bot.telegram_bot import TelegramBot

# Inherit callback class and overwrite on_epoch_end method
class TelegramSummary(Callback):
    def __init__(self):
        self.telegram_bot = TelegramBot()

    def on_train_begin(self, logs={}):
        answer = self.telegram_bot.send("Training started..........")
        return

    def on_train_end(self, logs={}):
        answer = self.telegram_bot.send("Training finished..........")
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        print("Sending results to telegram")
        loss = "loss: " + str(round(logs.get('loss'), 2))
        acc = "acc:  " + str(round(logs.get('acc'), 2))
        val_loss = "val loss: " + str(round(logs.get('val_loss'), 2))
        val_acc = "val acc:  " + str(round(logs.get('val_acc'), 2))
        summary = loss + "  " + val_loss + "\n" + acc + "  " + val_acc

        answer = self.telegram_bot.send(summary)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

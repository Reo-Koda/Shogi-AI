from gmail import sendMail

def notify_result(epoch, train_loss, val_loss):
    sendMail.send_message(
        "学習終了",
        f"デスクトップを確認してください\nepoch {epoch + 1:>5}, train_loss {train_loss:.6f}, val_loss {val_loss:.6f}"
    )
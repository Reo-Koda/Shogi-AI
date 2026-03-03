from gmail.sendMail import send_message

def notify_result(to: str, epoch: int, train_loss: float, val_loss: float):
    send_message(
        to,
        "学習終了",
        f"デスクトップを確認してください\nepoch {epoch + 1:>5}, train_loss {train_loss:.6f}, val_loss {val_loss:.6f}"
    )

import os.path
import base64
from email.message import EmailMessage

from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv('.env') # 実行ファイルからの相対パス
MAILACCOUNT = os.getenv('MAILACCOUNT')

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def create_message(sender, to, subject, body):
  message = EmailMessage()
  message.set_content(body)
  message["To"] = to
  message["From"] = sender
  message["Subject"] = subject

  encoded_message = base64.urlsafe_b64encode(
    message.as_bytes()
  ).decode()

  return {"raw": encoded_message}

def send_message(subject, message, isMain=False, creds=None):
  if not isMain:
    if os.path.exists("token.json"):
      creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
      raise RuntimeError("token.json not found (run auth setup once)")
    
    if not creds.valid:
      if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open("token.json", "w") as token:
          token.write(creds.to_json())
      else:
        raise RuntimeError("Credentials invalid and cannot refresh (re-run auth setup)")
  
  try:
    # Call the Gmail API
    service = build("gmail", "v1", credentials=creds)

    message = create_message(
      sender="me",
      to=MAILACCOUNT,
      subject=subject,
      body=message
    )

    sent = service.users().messages().send(
      userId="me",
      body=message
    ).execute()

    print("Message Id:", sent["id"])

  except HttpError as error:
    print(f"An error occurred: {error}")

def main():
  """Shows basic usage of the Gmail API.
  Lists the user's Gmail labels.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    send_message("Refresh access token", "token.json created.", isMain=True, creds=creds)


  except HttpError as error:
    # TODO(developer) - Handle errors from gmail API.
    print(f"An error occurred: {error}")


if __name__ == "__main__":
  main()
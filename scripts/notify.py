import os
import requests

def send_slack_alert(message):
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        print("No Slack webhook set.")
        return

    payload = {"text": message}
    requests.post(webhook, json=payload)

def send_email_alert(subject, body):
    # You can use SendGrid, Mailgun, etc.
    print(f"Email Alert: {subject} - {body}")

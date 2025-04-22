import mlflow
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from slack_sdk import WebClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "sample-model"
REFERENCE_DATA_PATH = "tests/test_dataset.csv"
CURRENT_DATA_PATH = "tests/current_dataset.csv"
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_CHANNEL = "#mlflow-cicd"

def send_slack_notification(message):
    """Send notification to Slack."""
    client = WebClient(token=SLACK_TOKEN)
    try:
        client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
        print("Slack notification sent successfully.")
    except Exception as e:
        print(f"Failed to send Slack notification: {e}")

def monitor_drift():
    """Monitor for data and target drift using Evidently."""
    # Load datasets
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    current_data = pd.read_csv(CURRENT_DATA_PATH)

    # Initialize Evidently report
    drift_report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])

    # Run drift analysis
    drift_report.run(reference_data=reference_data, current_data=current_data)
    drift_detected = drift_report.as_dict()['metrics'][0]['result']['drift_detected']

    # Check for drift and notify
    if drift_detected:
        message = "Drift detected in model data or target! Check Evidently report for details."
        send_slack_notification(message)
        print("Drift detected.")
    else:
        print("No drift detected.")

if __name__ == "__main__":
    monitor_drift()
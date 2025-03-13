import cv2
import os
import time
import base64
import threading
import smtplib
from email.message import EmailMessage
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Set up Gemini AI
GOOGLE_API_KEY = ""  # 🔴 Replace with your Gemini API Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ✅ Email Configuration
EMAIL_SENDER = ""  # 🔴 Replace with your email
EMAIL_PASSWORD = ""  # 🔴 Replace with your email app password
EMAIL_RECEIVER = ""  # 🔴 Replace with recipient email

IMAGE_PATH = "latest_frame.jpg"  # ✅ Overwrites previous frame
SEND_INTERVAL = 5  # ✅ Send image every 5 seconds
last_sent_time = 0  # ✅ Tracks last AI request time


def send_email_alert(subject, body):
    """Sends an AI-generated email alert with the detected fire/smoke image."""
    if not os.path.exists(IMAGE_PATH):
        print("⚠️ Email alert skipped: No image file found.")
        return

    try:
        # ✅ Remove newline & carriage return from subject
        subject = subject.replace("\n", " ").replace("\r", "").strip()

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content(body)

        with open(IMAGE_PATH, "rb") as img_file:
            msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename="fire_alert.jpg")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print("📧 AI Email Alert Sent Successfully!")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def analyze_with_gemini():
    """Sends the latest frame to Gemini AI to check for fire or smoke."""
    if not os.path.exists(IMAGE_PATH):
        print("⚠️ No image available for analysis. Skipping AI check.")
        return

    try:
        with open(IMAGE_PATH, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": """
                Analyze the image and determine if fire or smoke is present.
                If fire or smoke is detected, generate a complete email automatically.
                - Write a **clear and urgent subject** (avoid long text).
                - Write a **professional but urgent email body**.
                - **Include emergency contact numbers** for Fire Department and Ambulance (local numbers for MUMBAI MAHARASHTRA).
                - If no fire or smoke is detected, simply respond with "No fire detected."
                """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = gemini_model.invoke([message])
        result = response.content.strip()
        print("🔍 AI Detection Result:", result)

        if "No fire detected" in result:
            print("✅ No fire detected. Skipping email alert.")
            return

        # ✅ Extract subject and body automatically
        lines = result.split("\n")
        subject = lines[0] if len(lines) > 0 else "Fire Alert!"
        body = "\n".join(lines[1:]) if len(lines) > 1 else "Possible fire detected. Take immediate action."

        # ✅ Send email alert
        send_email_alert(subject, body)

        # ✅ Safe file deletion check
        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)

    except Exception as e:
        print("⚠️ AI Analysis Skipped: Image could not be processed.")


def process_frame(frame):
    """Saves the latest frame every 5 seconds and starts AI analysis."""
    global last_sent_time
    current_time = time.time()

    if current_time - last_sent_time >= SEND_INTERVAL:
        last_sent_time = current_time
        cv2.imwrite(IMAGE_PATH, frame)  # ✅ Overwrite latest frame

        ai_thread = threading.Thread(target=analyze_with_gemini)
        ai_thread.daemon = True
        ai_thread.start()


def start_monitoring(video_file):
    """Reads video frames and monitors for fire/smoke."""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        process_frame(frame)

        cv2.imshow("🔥 Fire/Smoke Monitoring", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Monitoring Completed.")


if __name__ == "__main__":
    video_file = "fire.mp4"  # Change to 0 for live webcam
    start_monitoring(video_file)

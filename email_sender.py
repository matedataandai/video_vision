from email.message import EmailMessage
import smtplib
import os
from dotenv import load_dotenv
load_dotenv()


class EmailSender:
    def __init__(self):
        self.sender_email = os.getenv("EMAIL_ADDRESS")
        self.sender_password = os.getenv("EMAIL_PASSWORD")

    def send_email(self, receiver_email, court, unique_id,amount,duration):
        msg = EmailMessage()
        msg["From"] = self.sender_email
        msg["To"] = f'{receiver_email}, {self.sender_email}'
        msg["Subject"] = "Thank you for your booking - Northern Beaches Video Tennis Session"
        body = f"""Thank you for your booking with Northern Beaches Video Tennis Session!

Your booking details:
- Amount: ${amount:.2f} AUD
- Court: {court}
- Duration: {duration}
- Unique service: {unique_id}

You can watch your video at localhost:8501/?v={unique_id}

Feel free to reach out to us at 0406292441 or reply to this email if you have any questions or need further assistance.

Best regards,
The Northern Beaches Video Tennis Session Team"""
        
        msg.set_content(body)

        try:
          print("Connecting to Gmail SMTP server...")
          with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
          print("Email sent successfully via Gmail!")
        except Exception as e:
          print(f"Error: {e}")
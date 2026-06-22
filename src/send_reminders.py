"""
Standalone background reminder sender.
Checks PostgreSQL database every minute and sends email when reminder is due.
Uses SQLAlchemy to query the database directly.

Run this in a separate terminal / process / as a service:
python send_reminders.py

Optional: simulate a future time for testing:
python send_reminders.py "2026-03-03 08:23:00"
"""

import json
import time
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Set up fake datetime if provided
import sys
class OffsetDateTime(datetime):
    _offset = timedelta(0)
    @classmethod
    def now(cls, tz=None):
        return super().now(tz) + cls._offset

datetime = OffsetDateTime

if len(sys.argv) > 1:
    fake = OffsetDateTime.fromisoformat(sys.argv[1].replace(" ", "T"))
    OffsetDateTime._offset = fake - OffsetDateTime.now()

load_dotenv()

# Database imports
from database import get_session, ReminderDB, TaskDB, MedicationDB

# ──────────────────────────────────────────────
#           CONFIGURATION (from .env)
# ──────────────────────────────────────────────
SENDER_EMAIL    = os.getenv("REMINDER_SENDER_EMAIL")          # e.g. yourname@gmail.com
APP_PASSWORD    = os.getenv("REMINDER_SENDER_APP_PASSWORD")   # Gmail app password (16 chars)
RECIPIENT_EMAIL = os.getenv("REMINDER_RECIPIENT_EMAIL")       # who receives reminders

SMTP_SERVER     = "smtp.gmail.com"
SMTP_PORT       = 465

USER_ID = "2"  # The user ID to send reminders for
CHECK_INTERVAL_SECONDS = 60   # Check every 1 minute
GRACE_PERIOD_MINUTES   = 15    # consider reminder "due" up to 15 min in future too

# ──────────────────────────────────────────────


def send_email(subject: str, body: str) -> bool:
    """Send an email reminder."""
    if not SENDER_EMAIL or not APP_PASSWORD or not RECIPIENT_EMAIL:
        print("⚠ Email config missing. Set REMINDER_SENDER_EMAIL, REMINDER_SENDER_APP_PASSWORD, REMINDER_RECIPIENT_EMAIL")
        return False
    
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = SENDER_EMAIL
            msg["To"] = RECIPIENT_EMAIL
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            server.send_message(msg)
            print(f"✓ Email sent to {RECIPIENT_EMAIL}: {subject}")
            return True
    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        return False


def get_pending_reminders_and_medications():
    """Get pending reminders and medication reminders from PostgreSQL database."""
    session = get_session()
    try:
        now = datetime.now()
        future = now + timedelta(minutes=GRACE_PERIOD_MINUTES)
        
        # Get pending task reminders from database
        reminders = ReminderDB.get_pending_reminders(session, USER_ID, window_minutes=GRACE_PERIOD_MINUTES)
        
        # Get pending tasks (overdue and due soon)
        all_tasks = TaskDB.get_user_tasks(session, USER_ID, completed=False)
        pending_tasks = []
        for task in all_tasks:
            if task.due_date and task.due_date <= future and task.due_date >= now:
                pending_tasks.append(task)
        
        # Get active medications
        active_meds = MedicationDB.get_user_medications(session, USER_ID, active=True)
        
        return reminders, pending_tasks, active_meds
    finally:
        session.close()


def process_reminders():
    """Check and send pending reminders."""
    session = get_session()
    try:
        reminders, pending_tasks, medications = get_pending_reminders_and_medications()
        
        if not reminders and not pending_tasks and not medications:
            print(f"[{datetime.now().isoformat()}] No pending reminders")
            return
        
        # Process task reminders
        for reminder in reminders:
            subject = f"Reminder: {reminder.title}"
            body = f"Time to: {reminder.title}\n\nScheduled for: {reminder.scheduled_time}"
            
            if reminder.description:
                body += f"\n\nDetails: {reminder.description}"
            
            send_email(subject, body)
            
            # Mark as sent
            ReminderDB.mark_reminder_sent(session, reminder.id)
        
        # Process task due dates
        for task in pending_tasks:
            subject = f"Task Reminder: {task.text}"
            body = f"This task is due: {task.text}\n\nDue date: {task.due_date}"
            
            if task.description:
                body += f"\n\nDetails: {task.description}"
            
            send_email(subject, body)
            
            # Create a reminder record for tracking
            reminder_id = f"task_{task.id}"
            existing = ReminderDB.get_reminder(session, reminder_id)
            if not existing:
                ReminderDB.create_reminder(
                    session=session,
                    user_id=USER_ID,
                    reminder_id=reminder_id,
                    title=f"Task: {task.text}",
                    scheduled_time=task.due_date,
                    task_id=task.id,
                    reminder_type="task"
                )
                ReminderDB.mark_reminder_sent(session, reminder_id)
        
        # Process medication reminders
        for med in medications:
            if med.frequency == "daily":
                subject = f"Medication Reminder: {med.name}"
                body = f"Time to take: {med.name}"
                
                if med.dosage:
                    body += f"\nDosage: {med.dosage}"
                if med.frequency:
                    body += f"\nFrequency: {med.frequency}"
                if med.notes:
                    body += f"\nNotes: {med.notes}"
                
                send_email(subject, body)
        
        print(f"✓ Processed {len(reminders)} reminders, {len(pending_tasks)} tasks, {len(medications)} medications")
    finally:
        session.close()


def main():
    """Main loop - check for reminders periodically."""
    print(f"Starting reminder sender for user {USER_ID}")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS}s")
    print(f"Email config: {SENDER_EMAIL} -> {RECIPIENT_EMAIL}")
    
    if not SENDER_EMAIL or not APP_PASSWORD or not RECIPIENT_EMAIL:
        print("\n⚠ Warning: Email not configured. Reminders will not be sent.")
        print("Set environment variables:")
        print("  REMINDER_SENDER_EMAIL")
        print("  REMINDER_SENDER_APP_PASSWORD")
        print("  REMINDER_RECIPIENT_EMAIL")
    
    print("\nStarting reminder check loop...\n")
    
    while True:
        try:
            process_reminders()
        except Exception as e:
            print(f"✗ Error: {e}")
        
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

"""
PostgreSQL database models and connection for tasks and reminders.
Uses SQLAlchemy with async support for efficient database operations.

Follows LangChain best practices for persistent storage:
https://docs.langchain.com/oss/python/langgraph/add-memory#database-management
"""

import os
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Integer, Text, select
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/mini_proj?sslmode=disable"
)

# Convert to SQLAlchemy dialect URIs
# Use postgresql+psycopg for sync (psycopg v3) and postgresql+asyncpg for async
SYNC_DB_URI = DB_URI.replace("postgresql://", "postgresql+psycopg://") if "postgresql://" in DB_URI else DB_URI
ASYNC_DB_URI = DB_URI.replace("postgresql://", "postgresql+asyncpg://") if "postgresql://" in DB_URI else DB_URI

# Create base class for models
Base = declarative_base()

# Global engine and session factory
_engine = None
_async_engine = None
_session_factory = None


class Task(Base):
    """Task model for storing user tasks in PostgreSQL."""
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    text = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    due_date = Column(DateTime, nullable=True, index=True)
    completed = Column(Boolean, default=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    priority = Column(String, default="medium")  # low, medium, high
    tags = Column(Text, nullable=True)  # Comma-separated tags
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "text": self.text,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed": self.completed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "priority": self.priority,
            "tags": self.tags.split(",") if self.tags else [],
        }


class Reminder(Base):
    """Reminder model for storing reminders in PostgreSQL."""
    __tablename__ = "reminders"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    task_id = Column(String, index=True, nullable=True)  # Reference to task if applicable
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    scheduled_time = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    sent = Column(Boolean, default=False, index=True)
    sent_at = Column(DateTime, nullable=True)
    reminder_type = Column(String, default="task")  # task, medication, custom
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "sent": self.sent,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "reminder_type": self.reminder_type,
        }


class Medication(Base):
    """Medication model for tracking medications."""
    __tablename__ = "medications"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    dosage = Column(String, nullable=True)
    frequency = Column(String, nullable=True)  # daily, twice_daily, weekly, etc.
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)
    active = Column(Boolean, default=True, index=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "active": self.active,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Synchronous database functions (for setup and initialization)
def init_db_sync():
    """Initialize database schema (synchronous)."""
    global _engine
    
    if _engine is None:
        _engine = create_engine(
            SYNC_DB_URI, 
            echo=False,
            pool_pre_ping=True,  # Test connections before using them
            pool_recycle=3600    # Recycle connections after 1 hour
        )
    
    Base.metadata.create_all(_engine)
    print("✓ Database schema initialized")


def get_session() -> Session:
    """Get a synchronous database session."""
    global _engine, _session_factory
    
    if _engine is None:
        _engine = create_engine(
            SYNC_DB_URI,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )
    
    if _session_factory is None:
        from sqlalchemy.orm import sessionmaker
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    
    return _session_factory()


def close_db_sync():
    """Close synchronous database connection."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


# Asynchronous database functions (for FastAPI)
async def init_db_async():
    """Initialize database schema (asynchronous)."""
    global _async_engine
    
    if _async_engine is None:
        _async_engine = create_async_engine(ASYNC_DB_URI, echo=False)
    
    async with _async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Async database schema initialized")


async def get_async_session() -> AsyncSession:
    """Get an asynchronous database session."""
    global _async_engine, _session_factory
    
    if _async_engine is None:
        _async_engine = create_async_engine(ASYNC_DB_URI, echo=False)
    
    if _session_factory is None:
        _session_factory = async_sessionmaker(_async_engine, class_=AsyncSession, expire_on_commit=False)
    
    async with _session_factory() as session:
        yield session


async def close_db_async():
    """Close asynchronous database connection."""
    global _async_engine
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None


# CRUD Operations (synchronous for background tasks)
class TaskDB:
    """Database operations for tasks."""
    
    @staticmethod
    def create_task(session: Session, user_id: str, task_id: str, text: str, 
                   due_date: Optional[datetime] = None, priority: str = "medium") -> Task:
        """Create a new task."""
        task = Task(
            id=task_id,
            user_id=user_id,
            text=text,
            due_date=due_date,
            priority=priority,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        return task
    
    @staticmethod
    def get_task(session: Session, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return session.query(Task).filter(Task.id == task_id).first()
    
    @staticmethod
    def get_user_tasks(session: Session, user_id: str, completed: Optional[bool] = None) -> List[Task]:
        """Get all tasks for a user."""
        query = session.query(Task).filter(Task.user_id == user_id)
        if completed is not None:
            query = query.filter(Task.completed == completed)
        return query.order_by(Task.created_at.desc()).all()
    
    @staticmethod
    def update_task(session: Session, task_id: str, **kwargs) -> Optional[Task]:
        """Update a task."""
        task = TaskDB.get_task(session, task_id)
        if not task:
            return None
        
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        session.commit()
        session.refresh(task)
        return task
    
    @staticmethod
    def delete_task(session: Session, task_id: str) -> bool:
        """Delete a task."""
        task = TaskDB.get_task(session, task_id)
        if not task:
            return False
        
        session.delete(task)
        session.commit()
        return True
    
    @staticmethod
    def mark_task_complete(session: Session, task_id: str) -> Optional[Task]:
        """Mark a task as completed."""
        return TaskDB.update_task(session, task_id, completed=True, completed_at=datetime.utcnow())


class ReminderDB:
    """Database operations for reminders."""
    
    @staticmethod
    def create_reminder(session: Session, user_id: str, reminder_id: str, title: str, 
                       scheduled_time: datetime, task_id: Optional[str] = None,
                       reminder_type: str = "task") -> Reminder:
        """Create a new reminder."""
        reminder = Reminder(
            id=reminder_id,
            user_id=user_id,
            title=title,
            scheduled_time=scheduled_time,
            task_id=task_id,
            reminder_type=reminder_type,
        )
        session.add(reminder)
        session.commit()
        session.refresh(reminder)
        return reminder
    
    @staticmethod
    def get_reminder(session: Session, reminder_id: str) -> Optional[Reminder]:
        """Get a reminder by ID."""
        return session.query(Reminder).filter(Reminder.id == reminder_id).first()
    
    @staticmethod
    def get_user_reminders(session: Session, user_id: str, sent: Optional[bool] = None) -> List[Reminder]:
        """Get all reminders for a user."""
        query = session.query(Reminder).filter(Reminder.user_id == user_id)
        if sent is not None:
            query = query.filter(Reminder.sent == sent)
        return query.order_by(Reminder.scheduled_time.asc()).all()
    
    @staticmethod
    def get_pending_reminders(session: Session, user_id: str, window_minutes: int = 60) -> List[Reminder]:
        """Get reminders that are due within the given window."""
        from sqlalchemy import and_
        now = datetime.utcnow()
        future = datetime.utcnow() + __import__('datetime').timedelta(minutes=window_minutes)
        
        return session.query(Reminder).filter(
            and_(
                Reminder.user_id == user_id,
                Reminder.sent == False,
                Reminder.scheduled_time <= future,
                Reminder.scheduled_time >= now,
            )
        ).order_by(Reminder.scheduled_time.asc()).all()
    
    @staticmethod
    def mark_reminder_sent(session: Session, reminder_id: str) -> Optional[Reminder]:
        """Mark a reminder as sent."""
        reminder = ReminderDB.get_reminder(session, reminder_id)
        if not reminder:
            return None
        
        reminder.sent = True
        reminder.sent_at = datetime.utcnow()
        session.commit()
        session.refresh(reminder)
        return reminder
    
    @staticmethod
    def delete_reminder(session: Session, reminder_id: str) -> bool:
        """Delete a reminder."""
        reminder = ReminderDB.get_reminder(session, reminder_id)
        if not reminder:
            return False
        
        session.delete(reminder)
        session.commit()
        return True


class MedicationDB:
    """Database operations for medications."""
    
    @staticmethod
    def create_medication(session: Session, user_id: str, med_id: str, name: str,
                         dosage: Optional[str] = None, frequency: Optional[str] = None) -> Medication:
        """Create a new medication."""
        med = Medication(
            id=med_id,
            user_id=user_id,
            name=name,
            dosage=dosage,
            frequency=frequency,
        )
        session.add(med)
        session.commit()
        session.refresh(med)
        return med
    
    @staticmethod
    def get_user_medications(session: Session, user_id: str, active: bool = True) -> List[Medication]:
        """Get all medications for a user."""
        return session.query(Medication).filter(
            Medication.user_id == user_id,
            Medication.active == active
        ).order_by(Medication.created_at.desc()).all()
    
    @staticmethod
    def update_medication(session: Session, med_id: str, **kwargs) -> Optional[Medication]:
        """Update a medication."""
        med = session.query(Medication).filter(Medication.id == med_id).first()
        if not med:
            return None
        
        for key, value in kwargs.items():
            if hasattr(med, key):
                setattr(med, key, value)
        
        session.commit()
        session.refresh(med)
        return med

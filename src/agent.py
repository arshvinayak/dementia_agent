from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import uuid
import os
import asyncio
import sys
from datetime import datetime, timedelta
import time
from sqlalchemy.orm import Session

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Import store and database
from store import get_store, close_store
from database import init_db_sync, get_session, close_db_sync, Task, Reminder, Medication, TaskDB, ReminderDB, MedicationDB

# Global references
store = None
agent = None
user_id = "2"
namespace = (user_id, "memories")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: Initialize store and agent on startup, cleanup on shutdown.
    """
    global store, agent
    
    # Startup
    print("\n[Startup] Initializing database...")
    init_db_sync()
    
    print("[Startup] Initializing LangGraph store...")
    store = get_store()
    
    tools = [
        create_manage_memory_tool(namespace), 
        create_search_memory_tool(namespace)
    ]
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=1.0)
    agent = create_agent(model, 
                        tools=tools, 
                        store=store, 
                        system_prompt="You are a helpful dementia patient assistant. help me with reminding of the daily tasks.")
    
    print("✓ App started - Store, database, and agent ready\n")
    
    yield  # App is running
    
    # Shutdown
    close_store()
    close_db_sync()
    print("✓ App shutdown - Store and database connections closed")

# FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# CORS Configuration - Allow frontend at localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# small helper for agent.invoke running off the event loop
async def safe_agent_invoke(prompt: str):
    try:
        if getattr(sys, "is_finalizing", lambda: False)():
            return "Task saved."
        def invoke():
            try:
                rs = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                return rs
            except Exception:
                return None
        result_state = await asyncio.to_thread(invoke)
        if not result_state:
            return "Task saved."
        # try to extract content
        try:
            return result_state["messages"][-1].content
        except Exception:
            return "Task saved."
    except RuntimeError:
        return "Task saved."
    except Exception:
        return "Task saved."

# API models
class TaskIn(BaseModel):
    """
    TaskIn model for incoming task requests.

    This Pydantic BaseModel represents the input data structure for a task.
    It is used to validate and serialize incoming task data, ensuring that
    the required 'text' field is present and correctly typed as a string.

    Attributes:
        text (str): The task description or input text content.
        date (str | None): Optional date in YYYY-MM-DD format.
        time (str | None): Optional time in HH:MM format.
    """
    text: str
    date: str | None = None
    time: str | None = None

class PromptIn(BaseModel):
    text: str

class MedicationIn(BaseModel):
    """
    Model for incoming medication requests.
    
    Attributes:
        name (str): Name of the medication.
        times (list[str]): Times to take medication (e.g., ["08:00", "14:00"]).
        frequency (str): "daily" or "custom".
        days (list[str]): Days to take medication (e.g., ["Monday", "Wednesday"]).
    """
    name: str
    times: list[str]
    frequency: str = "daily"
    days: list[str] | None = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Dementia Assistant API", "status": "running"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Backend API is running"}


@app.post("/api/tasks")
async def api_add_task(payload: TaskIn):
    """Create a new task."""
    try:
        if not payload.text or not payload.text.strip():
            raise HTTPException(status_code=400, detail="text required")
        
        # Construct full text with date and time if provided
        full_text = payload.text.strip()
        scheduled = None
        
        if payload.date or payload.time:
            datetime_str = ""
            if payload.date and payload.time:
                datetime_str = f" at {payload.time} on {payload.date}"
            elif payload.date:
                datetime_str = f" on {payload.date}"
            elif payload.time:
                datetime_str = f" at {payload.time}"
            full_text += datetime_str
            
            # Construct scheduled datetime from explicit date/time fields
            try:
                if payload.date and payload.time:
                    scheduled = datetime.fromisoformat(f"{payload.date}T{payload.time}")
                elif payload.date:
                    scheduled = datetime.fromisoformat(f"{payload.date}T00:00")
            except ValueError as e:
                # If date format is invalid, leave scheduled as None
                print(f"Date parsing warning: {e}")
                pass

        task_id = str(uuid.uuid4())
        
        # Store task in PostgreSQL database
        session = get_session()
        try:
            task = TaskDB.create_task(
                session=session,
                user_id=user_id,
                task_id=task_id,
                text=full_text,
                due_date=scheduled,
                priority="medium"
            )
            
            # Also store in LangGraph for agent memory (optional)
            if store:
                try:
                    memory = {
                        "id": task_id,
                        "task": full_text,
                        "text": full_text,
                        "scheduled": scheduled.isoformat() if scheduled else None,
                        "created_at": datetime.now().isoformat()
                    }
                    store.put(namespace, task_id, memory)
                except Exception as e:
                    print(f"Warning: Could not store in LangGraph: {e}")
            
            return {"task_id": task_id, "reply": "Task added successfully."}
        except Exception as e:
            print(f"Error creating task: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in api_add_task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/prompt")
async def api_prompt(payload: PromptIn):
    """Send a prompt to the agent."""
    try:
        if not payload.text or not payload.text.strip():
            raise HTTPException(status_code=400, detail="text required")
        
        try:
            start_time = time.time()
            result = agent.invoke(
                {"messages": [{"role": "user", "content": payload.text}]}
            )
            end_time = time.time()
            print(f"Agent latency: {end_time - start_time:.2f}s")
            
            msg = result["messages"][-1].content
            return {"reply": msg}
        except Exception as e:
            print(f"Error invoking agent: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process prompt: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in api_prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/reminders")
async def api_check_reminders(window: int = Query(60, gt=0)):
    """Get reminders from PostgreSQL database."""
    try:
        session = get_session()
        try:
            # Get pending reminders from database
            pending_reminders = ReminderDB.get_pending_reminders(session, user_id, window)
            
            # Get all tasks from database
            all_tasks = TaskDB.get_user_tasks(session, user_id)
            
            now = datetime.now()
            upcoming = []
            overdue = []
            future_all = []
            unscheduled = []
            all_tasks_list = []
            
            # Process tasks
            for task in all_tasks:
                task_dict = {
                    "id": task.id,
                    "text": task.text,
                    "scheduled": task.due_date.isoformat() if task.due_date else None,
                    "created_at": task.created_at.isoformat() if task.created_at else None
                }
                all_tasks_list.append(task_dict)
                
                if not task.due_date:
                    unscheduled.append(task.text)
                    continue
                
                if task.due_date <= now:
                    overdue.append({"text": task.text, "at": task.due_date.isoformat()})
                else:
                    future_all.append({"text": task.text, "at": task.due_date.isoformat(), "dt": task.due_date})
                    if task.due_date <= now + timedelta(minutes=window):
                        upcoming.append({"text": task.text, "at": task.due_date.isoformat()})
            
            # Add pending reminders to upcoming
            for reminder in pending_reminders:
                upcoming.append({"text": reminder.title, "at": reminder.scheduled_time.isoformat()})
            
            # Sort all_tasks by created_at descending
            try:
                all_tasks_sorted = sorted(
                    all_tasks_list,
                    key=lambda x: x.get("created_at") or "",
                    reverse=True
                )
            except Exception:
                all_tasks_sorted = all_tasks_list
            
            if overdue or upcoming:
                return {"overdue": overdue, "upcoming": upcoming, "all": all_tasks_sorted}
            
            # fallback
            if future_all:
                next_item = min(future_all, key=lambda x: x["dt"])
                return {"message": f"No reminders in window. Next: {next_item['text']} at {next_item['at']}", "all": all_tasks_sorted}
            
            if unscheduled:
                return {"message": "No scheduled reminders. Unscheduled tasks: " + "; ".join(unscheduled), "all": all_tasks_sorted}
            
            return {"message": "No reminders scheduled.", "all": all_tasks_sorted}
        finally:
            session.close()
    except Exception as e:
        print(f"Error in api_check_reminders: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get reminders: {str(e)}")

# new endpoint to delete a task by id
@app.delete("/api/tasks/{task_id}")
async def api_delete_task(task_id: str):
    """Delete a task from database."""
    try:
        session = get_session()
        try:
            # Delete from database
            success = TaskDB.delete_task(session, task_id)
            if not success:
                raise HTTPException(status_code=404, detail="Task not found")
            
            # Also delete from store
            try:
                store.delete(namespace, task_id)
            except Exception as e:
                print(f"Warning: Could not delete from store: {e}")
            
            return {"deleted": True}
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")


# ──────────────────────────────────────────────
#          MEDICATION ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/api/medications")
async def api_add_medication(payload: MedicationIn):
    """Add a new medication to database."""
    try:
        if not payload.name or not payload.name.strip():
            raise HTTPException(status_code=400, detail="name required")
        
        if not payload.times or len(payload.times) == 0:
            raise HTTPException(status_code=400, detail="at least one time required")
        
        med_id = str(uuid.uuid4())
        session = get_session()
        try:
            # Store medication in database
            medication = MedicationDB.create_medication(
                session=session,
                user_id=user_id,
                med_id=med_id,
                name=payload.name.strip(),
                frequency=payload.frequency or "daily"
            )
            
            # Also store in LangGraph for agent memory (optional)
            if store:
                try:
                    memory = {
                        "id": med_id,
                        "medicine": payload.name.strip(),
                        "name": payload.name.strip(),
                        "times": payload.times,
                        "days": payload.days or [],
                        "frequency": payload.frequency or "daily",
                        "created_at": datetime.now().isoformat()
                    }
                    store.put(namespace, med_id, memory)
                except Exception as e:
                    print(f"Warning: Could not store in LangGraph: {e}")
            
            return {"med_id": med_id, "reply": f"Medication '{payload.name}' added successfully."}
        except Exception as e:
            print(f"Error creating medication: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to add medication: {str(e)}")
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in api_add_medication: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/medications")
async def api_get_medications():
    """Get all medications for the user from database."""
    try:
        session = get_session()
        try:
            # Get medications from database
            medications = MedicationDB.get_user_medications(session, user_id, active=True)
            
            meds_list = [med.to_dict() for med in medications]
            return {"medications": meds_list}
        finally:
            session.close()
    except Exception as e:
        print(f"Error retrieving medications: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get medications: {str(e)}")


@app.delete("/api/medications/{med_id}")
async def api_delete_medication(med_id: str):
    """Delete a medication from database."""
    try:
        session = get_session()
        try:
            # Mark medication as inactive instead of deleting
            medication = MedicationDB.update_medication(session, med_id, active=False)
            if not medication:
                raise HTTPException(status_code=404, detail="Medication not found")
            
            # Also delete from store
            try:
                store.delete(namespace, med_id)
            except Exception as e:
                print(f"Warning: Could not delete from store: {e}")
            
            return {"deleted": True}
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting medication: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete medication: {str(e)}")


if __name__ == "__main__":
    import socket
    import errno

    HOST = os.getenv("HOST", "0.0.0.0")
    START_PORT = int(os.getenv("PORT", "8000"))
    MAX_PORT = START_PORT + 1000

    def _is_port_free(host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            s.close()
            return True
        except OSError:
            try:
                s.close()
            except Exception:
                pass
            return False

    chosen_port = None
    if _is_port_free(HOST, START_PORT):
        chosen_port = START_PORT
    else:
        for p in range(START_PORT + 1, MAX_PORT + 1):
            if _is_port_free(HOST, p):
                chosen_port = p
                break

    if chosen_port is None:
        raise RuntimeError(f"No free port found in range {START_PORT}-{MAX_PORT}")

    print(f"Starting server on http://{HOST}:{chosen_port} (requested PORT={START_PORT})")
    uvicorn.run("agent:app", host=HOST, port=chosen_port, reload=False)


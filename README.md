# Employee Attendance & Analytics API

A small attendance backend built with Python, FastAPI and MongoDB (PyMongo).
All application code is in `app/main.py`.

## Features
- Employees: create (with validation) and list (paginated)
- Attendance: punch-in, punch-out, list with filters
- Manual corrections (PATCH) with an audit history
- Analytics (MongoDB aggregation pipelines): employee monthly summary,
  department summary, late leaderboard, department trend
- `/admin/explain/{endpoint}` shows MongoDB's execution plan for each query

## Requirements
- Python 3.11+
- MongoDB 6.0 or newer

## Run
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    export MONGO_URI="mongodb://localhost:27017"
    export MONGO_DB="attendance_db"
    uvicorn app.main:app --port 8000

Indexes are created automatically at startup.
Open http://localhost:8000/docs to try the endpoints.
`GET /health` returns 200 when MongoDB is reachable.

## Notes
- Times are stored in UTC; business rules (shift, late, overtime) use IST.
- Concurrent edits of the same record: the second request gets 409 and can retry.
- Design decisions are in `DECISIONS.md`; the review of the starter code is in `REVIEW.md`.

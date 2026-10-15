"""
Employee Attendance & Analytics API (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (real environment variables win over a local .env)

Layout of this file
  1. configuration, indexes
  2. time helpers and business rules R1-R5 (pure functions, no database access)
  3. request models and validation helpers
  4. employee / attendance endpoints
  5. analytics endpoints (MongoDB aggregation pipelines)
  6. admin explain endpoint
"""
import calendar
import json
import logging
import os
import re
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date as date_cls
from datetime import datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Optional

from bson import json_util
from bson.decimal128 import Decimal128
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

load_dotenv()  # never overrides variables that are already set in the environment

log = logging.getLogger("attendance")

# tz_aware=True: PyMongo hands back timezone-aware (UTC) datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]  # R6
OFF_STATUSES = ("ABSENT", "LEAVE")
GRACE_SECONDS = 10 * 60  # R2
OVERTIME_MIN_MINUTES = 30  # R3
HALF_DAY_LIMIT = Decimal("4.50")  # R5

Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]
MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
MonthQ = Annotated[str, Query(pattern=MONTH_PATTERN)]
OptMonthQ = Annotated[Optional[str], Query(pattern=MONTH_PATTERN)]
# Epoch milliseconds: strict int (no floats, no numeric strings), seconds-looking values rejected.
EpochMs = Annotated[int, Field(strict=True, ge=100_000_000_000, le=4_102_444_800_000)]

_indexes_ready = False


# --------------------------------------------------------------------------- #
# 1. Indexes
# --------------------------------------------------------------------------- #
def ensure_indexes() -> None:
    """Idempotent: create_index on an identical existing index is a no-op."""
    global _indexes_ready
    emp, logs = db.employees, db.attendance_logs
    # employees
    emp.create_index([("emp_code", 1)], unique=True, name="emp_code_unique")
    emp.create_index([("department", 1), ("emp_code", 1)], name="dept_emp_code")
    emp.create_index([("joined_on", 1)], name="joined_on")
    emp.create_index([("department", 1), ("joined_on", 1)], name="dept_joined_on")
    # attendance_logs
    logs.create_index([("emp_code", 1), ("date", 1)], unique=True, name="emp_date_unique")
    logs.create_index([("date", -1), ("emp_code", 1)], name="date_desc_emp")
    logs.create_index([("status", 1), ("date", -1), ("emp_code", 1)], name="status_date_emp")
    logs.create_index([("emp_code", 1), ("punch_in", -1)], name="emp_punch_in")
    _indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
    except Exception:  # MongoDB may still be starting; /health retries
        log.exception("index creation failed at startup")
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 2. Time helpers and business rules (pure functions)
# --------------------------------------------------------------------------- #
def ms_to_dt(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(seconds=ms // 1000)


def dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:  # defensive: BSON dates are always UTC
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def now_ms_precision() -> datetime:
    n = datetime.now(UTC)
    return n.replace(microsecond=n.microsecond // 1000 * 1000)  # BSON keeps milliseconds


def round_half_up(x: Any, places: int) -> Decimal:
    if isinstance(x, Decimal128):
        x = x.to_decimal()
    d = x if isinstance(x, Decimal) else Decimal(str(x))
    return d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def hm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


def is_overnight(emp: dict) -> bool:
    return hm(emp["shift_end"]) <= hm(emp["shift_start"])


def attendance_date(punch_in: datetime, emp: dict) -> date_cls:
    """R1: IST calendar date of the punch-in; early punch-ins of an overnight shift belong to yesterday."""
    local = punch_in.astimezone(IST)
    day = local.date()
    if is_overnight(emp) and local.replace(tzinfo=None).time() < hm(emp["shift_end"]):
        day -= timedelta(days=1)
    return day


def shift_window(emp: dict, day: date_cls) -> tuple[datetime, datetime]:
    """Shift start and end instants for the shift that starts on `day` (end is next day if overnight)."""
    start = datetime.combine(day, hm(emp["shift_start"]), tzinfo=IST)
    end = datetime.combine(day, hm(emp["shift_end"]), tzinfo=IST)
    if is_overnight(emp):
        end += timedelta(days=1)
    return start, end


def derive_fields(emp: dict, date_str: str, status: str, punch_in: Optional[datetime],
                  punch_out: Optional[datetime]) -> dict:
    """R2-R5. Takes whole-second aware datetimes. Returns the four derived fields."""
    if status not in PRESENCE or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    start, end = shift_window(emp, date_cls.fromisoformat(date_str))
    one_sec = timedelta(seconds=1)
    late_secs = (punch_in - start) // one_sec
    late = late_secs // 60 if late_secs > GRACE_SECONDS else 0  # strictly more than 10:00; floor from shift_start
    out = {"work_hours": None, "late_minutes": late, "overtime_minutes": 0, "half_day": False}
    if punch_out is not None:
        hours = round_half_up(Decimal((punch_out - punch_in) // one_sec) / Decimal(3600), 2)
        ot = ((punch_out - end) // one_sec) // 60
        out.update(
            work_hours=float(hours),
            overtime_minutes=ot if ot >= OVERTIME_MIN_MINUTES else 0,
            half_day=hours < HALF_DAY_LIMIT,
        )
    return out


def month_bounds(month: str) -> tuple[str, str]:
    y, m = int(month[:4]), int(month[5:])
    return f"{month}-01", f"{month}-{calendar.monthrange(y, m)[1]:02d}"


def count_weekdays(a: date_cls, b: date_cls) -> int:
    if a > b:
        return 0
    n = (b - a).days + 1
    weeks, rem = divmod(n, 7)
    extra = sum(1 for i in range(rem) if (a + timedelta(days=weeks * 7 + i)).weekday() < 5)
    return weeks * 5 + extra


# --------------------------------------------------------------------------- #
# 3. Models and validation helpers
# --------------------------------------------------------------------------- #
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HHMM = r"^([01]\d|2[0-3]):[0-5]\d$"


def unprocessable(where: str, name: str, msg: str) -> HTTPException:
    return HTTPException(422, detail=[{"loc": [where, name], "msg": msg, "type": "value_error"}])


def parse_date(value: str, name: str, where: str = "query") -> date_cls:
    if not DATE_RE.match(value):
        raise unprocessable(where, name, "must be YYYY-MM-DD")
    try:
        return date_cls.fromisoformat(value)
    except ValueError:
        raise unprocessable(where, name, "not a valid calendar date")


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM)
    shift_end: str = Field(default="18:30", pattern=HHMM)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _joined_on_is_a_date(cls, v: str) -> str:
        if not DATE_RE.match(v):
            raise ValueError("must be YYYY-MM-DD")
        date_cls.fromisoformat(v)  # raises ValueError for 2026-02-30
        return v

    @model_validator(mode="after")
    def _shift_differs(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc["created_at"]) if doc.get("created_at") else None,
    }


def _ms_if_dt(v: Any) -> Any:
    return dt_to_ms(v) if isinstance(v, datetime) else v


def record_out(doc: dict) -> dict:
    """Stored attendance document -> API shape. No _id; instants as epoch ms; legacy gaps filled."""
    history = []
    for h in doc.get("history") or []:
        history.append({
            "at": _ms_if_dt(h["at"]),
            "by": h.get("by"),
            "reason": h.get("reason"),
            "changes": {k: {"from": _ms_if_dt(c.get("from")), "to": _ms_if_dt(c.get("to"))}
                        for k, c in (h.get("changes") or {}).items()},
        })
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": _ms_if_dt(doc.get("punch_in")),
        "punch_out": _ms_if_dt(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# 4. Employees and attendance
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        if not _indexes_ready:
            ensure_indexes()
    except Exception:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_ms_precision()
    try:
        db.employees.insert_one(doc)  # the unique index on emp_code decides, not a prior find
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q: dict = {"department": department} if department else {}
    total = db.employees.count_documents(q)
    cursor = db.employees.find(q, {"_id": 0}).sort("emp_code", 1).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "employee not found")
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    day = attendance_date(ts, emp).isoformat()
    doc = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **derive_fields(emp, day, body.status, ts, None),
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date) index makes this race-safe
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "employee not found")
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    # Most recent record whose punch_in <= ts. Absent/leave records have a null punch_in and never match.
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, sort=[("punch_in", -1)]
    )
    if not rec:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    if ts <= rec["punch_in"]:
        raise HTTPException(422, "punched_at must be after punch_in")
    if ts - rec["punch_in"] > timedelta(hours=24):
        raise HTTPException(422, "punch_out is more than 24 hours after punch_in")
    fields = derive_fields(emp, rec["date"], rec["status"], rec["punch_in"], ts)
    # Compare-and-set: only succeeds while punch_out is still null, so exactly one concurrent caller wins.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": {"punch_out": ts, **fields}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return record_out(updated)


def attendance_filter(emp_code, date_from, date_to, status) -> dict:
    q: dict = {}
    if emp_code:
        q["emp_code"] = emp_code
    f = parse_date(date_from, "date_from") if date_from else None
    t = parse_date(date_to, "date_to") if date_to else None
    if f and t and f > t:
        raise unprocessable("query", "date_from", "date_from must not be after date_to")
    if f or t:
        q["date"] = {}
        if f:
            q["date"]["$gte"] = date_from
        if t:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    return q


ATTENDANCE_SORT = [("date", -1), ("emp_code", 1)]


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = attendance_filter(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(q) if q else db.attendance_logs.estimated_document_count()
    cursor = db.attendance_logs.find(q).sort(ATTENDANCE_SORT).skip((page - 1) * page_size).limit(page_size)
    return {"items": [record_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    day = parse_date(date, "date", "path")
    for f in ("status", "punch_in", "punch_out"):  # explicit null is not "omitted"
        if f in body.model_fields_set and getattr(body, f) is None:
            raise unprocessable("body", f, "must not be null")

    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "employee not found")
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not rec:
        raise HTTPException(404, "no record for that date")

    old_status, old_in, old_out = rec["status"], rec.get("punch_in"), rec.get("punch_out")
    new_status = body.status or old_status

    if new_status in OFF_STATUSES:
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, f"{new_status} clears punch times; do not send punch_in/punch_out")
        new_in = new_out = None
    else:
        new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else old_in
        new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else old_out
        if new_in is None:
            raise HTTPException(422, "a presence status requires a punch_in")
        if body.punch_in is not None and attendance_date(new_in, emp) != day:
            raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if new_out is not None and not (new_in < new_out <= new_in + timedelta(hours=24)):
            raise HTTPException(422, "punch_out must be after punch_in and within 24 hours")

    if (new_status, new_in, new_out) == (old_status, old_in, old_out):
        raise HTTPException(422, "request changes nothing")

    new_vals = {"status": new_status, "punch_in": new_in, "punch_out": new_out,
                **derive_fields(emp, date, new_status, new_in, new_out)}
    old_vals = {"status": old_status, "punch_in": old_in, "punch_out": old_out,
                "work_hours": rec.get("work_hours"), "late_minutes": rec.get("late_minutes") or 0,
                "overtime_minutes": rec.get("overtime_minutes") or 0, "half_day": bool(rec.get("half_day", False))}
    changes = {k: {"from": old_vals[k], "to": new_vals[k]} for k in new_vals if old_vals[k] != new_vals[k]}
    entry = {"at": now_ms_precision(), "by": body.regularized_by, "reason": body.reason, "changes": changes}

    # Optimistic concurrency: the update only applies if the record is exactly as we read it.
    # A concurrent edit changes history length (or the punch fields), so the loser matches nothing -> 409.
    flt: dict = {"_id": rec["_id"], "status": old_status, "punch_in": old_in, "punch_out": old_out}
    hist = rec.get("history")
    flt["history"] = {"$size": len(hist)} if isinstance(hist, list) else None
    updated = db.attendance_logs.find_one_and_update(
        flt, {"$set": new_vals, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "record was modified concurrently, retry")
    return record_out(updated)


# --------------------------------------------------------------------------- #
# 5. Analytics (MongoDB aggregation pipelines)
# --------------------------------------------------------------------------- #
# Expressions over a stored attendance document. `date` is a "YYYY-MM-DD" string, parsed as UTC midnight,
# so $dayOfWeek returns the calendar weekday (1=Sun ... 7=Sat); Mon-Fri = 2..6.
_DOW = {"$dayOfWeek": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d", "timezone": "UTC"}}}
_LATE = {"$ifNull": ["$late_minutes", 0]}
_IS_PRESENT = {"$in": ["$status", PRESENCE]}
_HALF = {"$eq": [{"$ifNull": ["$half_day", False]}, True]}
# present contribution of one record: 1, or 0.5 for a half day, only for presence status on Mon-Fri (R5-R7)
PRESENT_WEEKDAY = {"$cond": [{"$and": [_IS_PRESENT, {"$in": [_DOW, [2, 3, 4, 5, 6]]}]},
                             {"$cond": [_HALF, 0.5, 1]}, 0]}
# present contribution on any day (used by the trend, which counts weekend records too)
PRESENT_ANY_DAY = {"$cond": [_IS_PRESENT, {"$cond": [_HALF, 0.5, 1]}, 0]}
IS_LATE = {"$cond": [{"$gt": [_LATE, 0]}, 1, 0]}


def employee_monthly_pipeline(emp_code: str, first: str, last: str) -> list:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},
        {"$group": {
            "_id": None,
            "present_days": {"$sum": PRESENT_WEEKDAY},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": IS_LATE},
            "total_late_minutes": {"$sum": _LATE},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: MonthQ):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "employee not found")
    first, last = month_bounds(month)
    rows = list(db.attendance_logs.aggregate(employee_monthly_pipeline(emp_code, first, last)))
    s = rows[0] if rows else {}
    joined = date_cls.fromisoformat(emp["joined_on"])
    working_days = count_weekdays(max(date_cls.fromisoformat(first), joined), date_cls.fromisoformat(last))
    present = Decimal(str(s.get("present_days", 0)))
    pct = float(round_half_up(present / working_days * 100, 2)) if working_days > 0 else None
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": float(round_half_up(present, 2)),
        "leave_days": int(s.get("leave_days", 0)),
        "late_count": int(s.get("late_count", 0)),
        "total_late_minutes": int(s.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(s.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


def department_summary_pipeline(first: str, last: str, department: Optional[str]) -> tuple[list, str]:
    """Start from employees so people with zero logs still count (R9); join their month of logs in a sub-pipeline."""
    match: dict = {"joined_on": {"$lte": last}}
    hint = "joined_on"
    if department:
        match["department"] = department
        hint = "dept_joined_on"
    has_hours = {"$and": [_IS_PRESENT, {"$ne": [{"$ifNull": ["$work_hours", None]}, None]}]}
    pipeline = [
        {"$match": match},
        {"$lookup": {
            "from": "attendance_logs",
            "localField": "emp_code",
            "foreignField": "emp_code",  # served by the (emp_code, date) index
            "pipeline": [
                {"$match": {"date": {"$gte": first, "$lte": last}}},
                {"$group": {
                    "_id": None,
                    "present_days": {"$sum": PRESENT_WEEKDAY},
                    # Decimal128 keeps the sum of 2-decimal hours exact, so the final half-up rounding is right
                    "hours_sum": {"$sum": {"$cond": [has_hours, {"$toDecimal": "$work_hours"}, 0]}},
                    "hours_cnt": {"$sum": {"$cond": [has_hours, 1, 0]}},
                    "late_count": {"$sum": IS_LATE},
                    "total_late_minutes": {"$sum": _LATE},
                    "leave_count": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                    "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}},
                }},
            ],
            "as": "s",
        }},
        {"$unwind": {"path": "$s", "preserveNullAndEmptyArrays": True}},  # keep employees with no logs
        {"$group": {
            "_id": "$department",
            "headcount": {"$sum": 1},
            "present_days": {"$sum": "$s.present_days"},
            "hours_sum": {"$sum": "$s.hours_sum"},
            "hours_cnt": {"$sum": "$s.hours_cnt"},
            "late_count": {"$sum": "$s.late_count"},
            "total_late_minutes": {"$sum": "$s.total_late_minutes"},
            "leave_count": {"$sum": "$s.leave_count"},
            "on_duty_count": {"$sum": "$s.on_duty_count"},
        }},
        {"$sort": {"_id": 1}},
    ]
    return pipeline, hint


@app.get("/analytics/departments/summary")
def department_summary(month: MonthQ, department: Optional[str] = None):
    first, last = month_bounds(month)
    pipeline, hint = department_summary_pipeline(first, last, department)
    items = []
    for r in db.employees.aggregate(pipeline, hint=hint):
        cnt = int(r.get("hours_cnt", 0))
        hours_sum = r["hours_sum"].to_decimal() if isinstance(r["hours_sum"], Decimal128) else Decimal(str(r["hours_sum"]))
        items.append({
            "department": r["_id"],
            "headcount": r["headcount"],
            "present_days": float(round_half_up(r.get("present_days", 0), 2)),
            "avg_work_hours": float(round_half_up(hours_sum / cnt, 2)) if cnt else None,
            "late_count": int(r.get("late_count", 0)),
            "total_late_minutes": int(r.get("total_late_minutes", 0)),
            "leave_count": int(r.get("leave_count", 0)),
            "on_duty_count": int(r.get("on_duty_count", 0)),
        })
    return {"month": month, "items": items}


def late_leaderboard_pipeline(first: str, last: str, limit: int, department: Optional[str]) -> list:
    pipeline: list = [
        {"$match": {"date": {"$gte": first, "$lte": last}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": _LATE}, "late_count": {"$sum": IS_LATE}}},
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "e"}},
        {"$unwind": "$e"},  # logs of unknown employees disappear here
    ]
    if department:
        pipeline.append({"$match": {"e.department": department}})  # rank only inside the department
    pipeline += [
        # $rank = standard competition ranking (1, 2, 2, 4); computed BEFORE the limit
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$e.name", "department": "$e.department",
                      "total_late_minutes": 1, "late_count": 1}},
    ]
    return pipeline


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: MonthQ, limit: int = Query(10, ge=1, le=50), department: Optional[str] = None):
    first, last = month_bounds(month)
    rows = db.attendance_logs.aggregate(late_leaderboard_pipeline(first, last, limit, department),
                                        hint="date_desc_emp")
    items = [{"rank": int(r["rank"]), "emp_code": r["emp_code"], "name": r["name"], "department": r["department"],
              "total_late_minutes": int(r["total_late_minutes"]), "late_count": int(r["late_count"])} for r in rows]
    return {"month": month, "items": items}


def trend_inputs(department: str, frm: str, to: str) -> tuple[date_cls, date_cls, list]:
    f, t = parse_date(frm, "from"), parse_date(to, "to")
    if t < f:
        raise unprocessable("query", "to", "to must not be before from")
    if (t - f).days + 1 > 92:
        raise unprocessable("query", "to", "range must not exceed 92 days")
    emps = list(db.employees.find({"department": department}, {"_id": 0, "emp_code": 1, "joined_on": 1}))
    if not emps:
        raise HTTPException(404, "unknown department")
    return f, t, emps


def department_trend_pipeline(f: date_cls, t: date_cls, emps: list) -> list:
    codes = [e["emp_code"] for e in emps]
    joins = [{"d": d, "n": n} for d, n in sorted(Counter(e["joined_on"] for e in emps).items())]
    start = datetime.combine(f, time(0), tzinfo=UTC)
    stop = datetime.combine(t, time(0), tzinfo=UTC)
    return [
        {"$match": {"emp_code": {"$in": codes}, "date": {"$gte": f.isoformat(), "$lte": t.isoformat()}}},
        {"$group": {"_id": "$date", "present_count": {"$sum": PRESENT_ANY_DAY}, "late_count": {"$sum": IS_LATE}}},
        {"$addFields": {"d": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d", "timezone": "UTC"}}}},
        # gap filling: one row per calendar day. Upper bound is padded by a day and trimmed again, so it
        # does not matter whether the server treats the bound as inclusive or exclusive.
        {"$densify": {"field": "d", "range": {"step": 1, "unit": "day", "bounds": [start, stop + timedelta(days=1)]}}},
        {"$match": {"d": {"$lte": stop}}},
        {"$addFields": {
            "present_count": {"$ifNull": ["$present_count", 0]},
            "late_count": {"$ifNull": ["$late_count", 0]},
            "date": {"$dateToString": {"date": "$d", "format": "%Y-%m-%d"}},
            "is_working_day": {"$and": [{"$gte": [{"$dayOfWeek": "$d"}, 2]}, {"$lte": [{"$dayOfWeek": "$d"}, 6]}]},
        }},
        {"$addFields": {"headcount": {"$sum": {"$map": {
            "input": {"$literal": joins}, "as": "j",
            "in": {"$cond": [{"$lte": ["$$j.d", "$date"]}, "$$j.n", 0]},
        }}}}},
        {"$addFields": {"attendance_rate": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            {"$divide": ["$present_count", "$headcount"]}, None]}}},
        # $avg ignores nulls; window = current row + up to 6 earlier rows inside the range
        {"$setWindowFields": {"sortBy": {"d": 1},
                              "output": {"moving_avg_7d": {"$avg": "$attendance_rate", "window": {"documents": [-6, 0]}}}}},
        {"$project": {"_id": 0, "date": 1, "is_working_day": 1, "headcount": 1, "present_count": 1,
                      "late_count": 1, "attendance_rate": 1, "moving_avg_7d": 1}},
    ]


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    f, t, emps = trend_inputs(department, from_, to)
    items = []
    for r in db.attendance_logs.aggregate(department_trend_pipeline(f, t, emps)):
        rate = None
        if r.get("attendance_rate") is not None:  # recompute from exact counts so the half-up rounding is exact
            rate = float(round_half_up(Decimal(str(r["present_count"])) / Decimal(r["headcount"]), 4))
        avg = r.get("moving_avg_7d")
        items.append({
            "date": r["date"],
            "is_working_day": bool(r["is_working_day"]),
            "headcount": int(r["headcount"]),
            "present_count": float(r["present_count"]),
            "late_count": int(r["late_count"]),
            "attendance_rate": rate,
            "moving_avg_7d": float(round_half_up(avg, 4)) if avg is not None else None,
        })
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 6. Admin: explain
# --------------------------------------------------------------------------- #
def _need(**params) -> None:
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise HTTPException(422, detail=[{"loc": ["query", k], "msg": "field required", "type": "missing"}
                                         for k in missing])


@app.get("/admin/explain/{endpoint}")
def explain(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard",
                      "department_trend"],
    emp_code: Optional[str] = None,
    month: OptMonthQ = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Runs explain('executionStats') on the same query/pipeline the real endpoint builds."""
    if endpoint == "attendance_list":
        collection = "attendance_logs"
        body = {"filter": attendance_filter(emp_code, date_from, date_to, status),
                "sort": dict(ATTENDANCE_SORT), "skip": (page - 1) * page_size, "limit": page_size}
        cmd: dict = {"find": collection, **body}
    else:
        if endpoint == "employee_monthly":
            _need(emp_code=emp_code, month=month)
            first, last = month_bounds(month)
            collection, extra = "attendance_logs", {}
            pipeline = employee_monthly_pipeline(emp_code, first, last)
        elif endpoint == "department_summary":
            _need(month=month)
            first, last = month_bounds(month)
            collection = "employees"
            pipeline, hint = department_summary_pipeline(first, last, department)
            extra = {"hint": hint}
        elif endpoint == "late_leaderboard":
            _need(month=month)
            first, last = month_bounds(month)
            collection, extra = "attendance_logs", {"hint": "date_desc_emp"}
            pipeline = late_leaderboard_pipeline(first, last, limit, department)
        else:  # department_trend
            _need(department=department, **{"from": from_, "to": to})
            f, t, emps = trend_inputs(department, from_, to)
            collection, extra = "attendance_logs", {}
            pipeline = department_trend_pipeline(f, t, emps)
        cmd = {"aggregate": collection, "pipeline": pipeline, "cursor": {}, **extra}
    result = db.command({"explain": cmd, "verbosity": "executionStats"})
    return JSONResponse(json.loads(json_util.dumps({"endpoint": endpoint, "collection": collection,
                                                    "explain": result})))

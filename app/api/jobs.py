import uuid

# In-memory job store
_jobs: dict[str, dict] = {}

def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "message": "Job queued...",
        "chunks_created": None,
        "source": None
    }
    return job_id

def update_job(job_id: str, **kwargs):
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)

def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)
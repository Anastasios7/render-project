from fastapi import FastAPI
from insurance_logic import run_all

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "message": "Insurance API is running"}

@app.get("/run")
def run():
    return run_all()

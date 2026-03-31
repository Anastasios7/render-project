from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}

@app.get("/test")
def test():
    return {"message": "API working"}

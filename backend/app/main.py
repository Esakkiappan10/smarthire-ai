from fastapi import FastAPI

app = FastAPI(title="SmartHire AI API")

@app.get("/")
def root():
    return {"message": "SmartHire AI Backend is running 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

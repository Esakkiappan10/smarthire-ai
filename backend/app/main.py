from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SmartHire AI - Resume Relevance Check")

# Allow requests from frontend (optional, useful if you use Streamlit or React as frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "SmartHire AI Backend is running 🚀"})

@app.post("/evaluate_resume")
async def evaluate_resume(
    uploaded_file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    # Here you would process the file and job description
    # For now, return a dummy response like your Streamlit app
    if uploaded_file and jd_text:
        response = {
            "status": "success",
            "relevance_score": "85/100",
            "missing_skills": ["Docker", "Kubernetes"],
            "verdict": "High Suitability"
        }
    else:
        response = {
            "status": "error",
            "message": "Please upload a resume and enter a job description."
        }

    return JSONResponse(content=response)

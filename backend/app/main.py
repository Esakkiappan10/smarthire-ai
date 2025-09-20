from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
import aiofiles
from services.aimatchmaker import AIResumeEvaluator
from services.resumeparser import ResumeParser
import asyncio
import tempfile

asyncio.get_event_loop().set_exception_handler(lambda loop, context: None)

load_dotenv()

app = FastAPI(title="SmartHire AI API", version="1.0.0")

# Initialize services
resume_parser = ResumeParser()
ai_evaluator = AIResumeEvaluator()

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SmartHire AI API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_model": "loaded"}

@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and parse resume file"""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg']
        file_ext = os.path.splitext(file.filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported types: {', '.join(allowed_extensions)}"
            )
        
        # Save to a secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            file_path = tmp_file.name
        
        # Extract text
        extracted_text = resume_parser.extract_text(file_path)
        
        # Always clean extracted text
        extracted_text = resume_parser._clean_text(extracted_text)
        
        # Clean up temp file
        os.unlink(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing file: {str(e)}"
        )
    
@app.post("/api/evaluate")
async def evaluate_resume(data: dict):
    """Evaluate resume against job description"""
    try:
        resume_text = data.get('resume_text', '')
        jd_text = data.get('jd_text', '')
        
        if not resume_text or not jd_text:
            raise HTTPException(status_code=400, detail="Both resume_text and jd_text are required")
        
        # Perform evaluation
        evaluation_result = ai_evaluator.evaluate(resume_text, jd_text)
        
        return {
            "status": "success",
            "evaluation": evaluation_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/analyze-jd")
async def analyze_jd(data: dict):
    """Analyze job description to extract requirements"""
    try:
        jd_text = data.get('jd_text', '')
        if not jd_text:
            raise HTTPException(status_code=400, detail="jd_text is required")
        
        analysis = ai_evaluator.analyze_job_description(jd_text)
        
        return {
            "status": "success",
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/api/batch-evaluate")
async def batch_evaluate(data: dict):
    """Batch evaluate multiple resumes"""
    try:
        resumes = data.get('resumes', [])
        jd_text = data.get('jd_text', '')
        
        if not resumes or not jd_text:
            raise HTTPException(status_code=400, detail="resumes list and jd_text are required")
        
        results = []
        for idx, resume_text in enumerate(resumes):
            try:
                result = ai_evaluator.evaluate(resume_text, jd_text)
                results.append({
                    "resume_index": idx,
                    "evaluation": result
                })
            except Exception as e:
                results.append({
                    "resume_index": idx,
                    "error": str(e)
                })
        
        return {
            "status": "success",
            "results": results,
            "total_processed": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
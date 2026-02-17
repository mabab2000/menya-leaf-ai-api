from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import requests
import json
from urllib.parse import urlparse, unquote, parse_qs
from sqlalchemy import create_engine, text, bindparam, JSON
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI(title="Plant Disease Analyzer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database configuration (use provided DB URL by default)
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres.jwdhnfdgkokpyvrgmtxd:c01rJFqtYpcy8JU7@aws-1-eu-west-3.pooler.supabase.com:6543/postgres")
engine = create_engine(DB_URL)

def init_db():
    """Create result table if it doesn't exist"""
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS result (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                result JSONB,
                created_at TIMESTAMP DEFAULT now()
            )
            """
        ))

# Initialize DB on startup
@app.on_event("startup")
def on_startup():
    try:
        init_db()
        print("DB init: ensured 'result' table exists")
    except Exception as e:
        # surface the error so operator can run migration manually
        print(f"DB init failed on startup: {e}")
        print("You can run 'python migrate.py' to create the table.")

class AnalysisResponse(BaseModel):
    is_plant: bool
    message: str
    disease: str = None
    severity: str = None
    recommendations: list[str] = None
    affected_parts: list[str] = None

def encode_image(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')

def validate_image(file: UploadFile) -> bytes:
    """Validate and read image file"""
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Read and validate image
    contents = file.file.read()
    try:
        img = Image.open(BytesIO(contents))
        img.verify()
        return contents
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

def analyze_plant_image(image_base64: str) -> AnalysisResponse:
    """Analyze plant image using OpenAI Vision API"""
    
    prompt = """You are an expert plant pathologist. Analyze this image carefully and provide a detailed assessment.

First, determine if this image contains a plant. If it does NOT contain a plant, respond with:
{
  "is_plant": false,
  "message": "The uploaded image appears to be related to [describe what you see], not related to plants."
}

If the image DOES contain a plant, analyze it for diseases and respond with:
{
  "is_plant": true,
  "message": "Plant analysis completed successfully.",
  "disease": "[Name of disease if present, or 'Healthy' if no disease detected]",
  "severity": "[None/Mild/Moderate/Severe]",
  "affected_parts": ["list of affected plant parts like leaves, stems, roots, etc."],
  "recommendations": [
    "Specific treatment recommendation 1",
    "Specific treatment recommendation 2",
    "Prevention measure 1",
    "Prevention measure 2"
  ]
}

Be specific about:
- Disease identification (common and scientific names if applicable)
- Visible symptoms (spots, discoloration, wilting, etc.)
- Severity level
- Treatment options (organic and chemical)
- Prevention measures

Respond ONLY with valid JSON, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Parse the response safely and support different response shapes
        import json

        # Try to extract the content string from various possible shapes
        content_str = None
        try:
            # response may be a dict-like or object with attributes
            choices = None
            if hasattr(response, "choices"):
                choices = response.choices
            elif isinstance(response, dict):
                choices = response.get("choices")

            if choices and len(choices) > 0:
                first = choices[0]
                # message may be attribute or dict
                message = getattr(first, "message", None) if not isinstance(first, dict) else first.get("message")

                # message.content may be a string or a list of content blocks
                content = None
                if isinstance(message, dict):
                    content = message.get("content")
                else:
                    content = getattr(message, "content", None)

                if isinstance(content, str):
                    content_str = content
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            parts.append(item)
                    content_str = "".join(parts)

        except Exception:
            content_str = None

        # Fallback to stringifying the whole response
        if not content_str:
            try:
                content_str = str(response)
            except Exception:
                content_str = ""

        # Strip markdown code fences if present
        content_str = content_str.strip()
        if content_str.startswith("```json"):
            content_str = content_str[7:]  # Remove ```json
        elif content_str.startswith("```"):
            content_str = content_str[3:]  # Remove ```
        if content_str.endswith("```"):
            content_str = content_str[:-3]  # Remove trailing ```
        content_str = content_str.strip()

        # Attempt JSON parse
        try:
            result = json.loads(content_str)
            return AnalysisResponse(**result)
        except json.JSONDecodeError:
            # Provide truncated raw response to aid debugging
            preview = content_str[:2000]
            raise HTTPException(status_code=500, detail=f"Error parsing AI response. Raw response preview: {preview}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Disease Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Upload plant image for disease analysis",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status": "healthy",
        "openai_configured": api_key_present
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_plant(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Analyze uploaded plant image for diseases
    
    Args:
        file: Image file (jpg, jpeg, png, webp)
    
    Returns:
        Analysis results including disease identification and recommendations
    """
    
    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file"
        )
    
    # Validate and read image bytes
    image_bytes = validate_image(file)

    # First upload image to the file-vault API
    upload_url = "https://file-vault-ro9o.onrender.com/upload"
    try:
        # Use BytesIO so requests can stream the bytes
        from io import BytesIO
        files = {
            'file': (file.filename, BytesIO(image_bytes), file.content_type or 'application/octet-stream')
        }
        resp = requests.post(upload_url, headers={"accept": "application/json"}, files=files, timeout=30)
        resp.raise_for_status()
        upload_info = resp.json()
        image_path = upload_info.get('preview_url') or upload_info.get('path') or ''
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

    # Encode image to base64 for analysis (we keep using existing analyzer)
    image_base64 = encode_image(image_bytes)

    # Analyze image
    analysis = analyze_plant_image(image_base64)

    # Persist result in DB (bind JSON using appropriate type)
    try:
        with engine.begin() as conn:
            stmt = text("INSERT INTO result (user_id, image_path, result) VALUES (:user_id, :image_path, :result)")
            stmt = stmt.bindparams(bindparam('result', type_=JSON))
            conn.execute(
                stmt,
                {
                    'user_id': user_id,
                    'image_path': image_path,
                    'result': analysis.dict()
                }
            )
    except Exception as e:
        # Log but still return analysis
        print(f"DB insert failed: {e}")

    return analysis


def _extract_filename_from_path(image_path: str) -> str:
    """Try to extract a filename suitable for the preview API from a stored image_path.

    Handles either a raw filename (e.g., 'Screenshot 2026-02-13 194928.png') or
    a full signed URL returned earlier (extracts the last path segment).
    """
    if not image_path:
        return image_path
    try:
        parsed = urlparse(image_path)
        if parsed.scheme in ("http", "https"):
            # prefer last path segment
            last = parsed.path.split("/")[-1]
            if last:
                return unquote(last)
            # fallback: check for 'url' param containing the inner path
            qs = parse_qs(parsed.query)
            if "url" in qs and qs["url"]:
                inner = qs["url"][0]
                inner_p = urlparse(inner)
                return unquote(inner_p.path.split("/")[-1])
        # assume it's already a filename
        return image_path
    except Exception:
        return image_path


@app.get("/results/{user_id}")
def get_results(user_id: str):
    """Retrieve analysis results for a given `user_id` and return preview URLs.

    For each stored `image_path` we call the file-vault preview API with the
    extracted filename and return the `preview_url` along with the stored result.
    """
    rows = []
    try:
        with engine.connect() as conn:
            q = text("SELECT id, user_id, image_path, result, created_at FROM result WHERE user_id = :user_id ORDER BY created_at DESC")
            res = conn.execute(q, {"user_id": user_id})
            for r in res.fetchall():
                rows.append({
                    "id": r[0],
                    "user_id": r[1],
                    "image_path": r[2],
                    "result": r[3],
                    "created_at": r[4].isoformat() if r[4] is not None else None,
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")

    # For each row call preview API and replace image_path with preview_url when available
    preview_api = "https://file-vault-ro9o.onrender.com/preview"
    out = []
    for r in rows:
        filename = _extract_filename_from_path(r["image_path"]) if r.get("image_path") else ""
        preview_url = None
        if filename:
            try:
                resp = requests.get(preview_api, params={"path": filename}, headers={"accept": "application/json"}, timeout=10)
                if resp.ok:
                    data = resp.json()
                    preview_url = data.get("preview_url")
            except Exception:
                preview_url = None

        # If preview_url not available but stored image_path already looks like a URL, use it
        if not preview_url and r.get("image_path"):
            if str(r.get("image_path")).startswith("http"):
                preview_url = r.get("image_path")

        out.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "preview_url": preview_url,
            "result": r["result"],
            "created_at": r["created_at"],
        })

    return {"count": len(out), "items": out}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
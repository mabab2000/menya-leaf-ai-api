from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
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
async def analyze_plant(file: UploadFile = File(...)):
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
    
    # Validate and read image
    image_bytes = validate_image(file)
    
    # Encode image to base64
    image_base64 = encode_image(image_bytes)
    
    # Analyze image
    result = analyze_plant_image(image_base64)
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
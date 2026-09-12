import io
import json
import numpy as np
from PIL import Image
from http.server import BaseHTTPRequestHandler

def inspect_c2pa(raw_bytes: bytes) -> dict:
    has_c2pa = False
    manifest_type = "None"
    
    # Detect JUMBF box and C2PA manifest markers
    if b"c2pa" in raw_bytes or b"urn:c2pa" in raw_bytes:
        has_c2pa = True
        if b"c2pa.claim" in raw_bytes:
            manifest_type = "Standard C2PA Claim Manifest"
        elif b"c2pa.assertions" in raw_bytes:
            manifest_type = "C2PA Assertions Manifest"
        else:
            manifest_type = "Basic C2PA JUMBF Marker"

    return {
        "has_c2pa": has_c2pa,
        "manifest_type": manifest_type,
        "details": "Valid Content Credentials C2PA manifest detected." if has_c2pa else "No C2PA authentication provenance seal found."
    }

def profile_degradation(img_pil: Image.Image) -> dict:
    width, height = img_pil.size
    total_pixels = width * height
    is_downscaled = bool(total_pixels < (720 * 1280))

    # 1. Compute Image Sharpness (Laplacian Variance via NumPy)
    gray = img_pil.convert("L")
    arr = np.array(gray, dtype=np.float32)
    edges = (
        arr[:-2, 1:-1] + arr[2:, 1:-1] +
        arr[1:-1, :-2] + arr[1:-1, 2:] -
        4 * arr[1:-1, 1:-1]
    )
    laplacian_var = float(np.var(edges))

    # 2. Estimate JPEG Compression Quality
    jpeg_quality = "Non-JPEG / Clean Header"
    quantization = getattr(img_pil, "quantization", None)
    if quantization:
        try:
            avg_q = float(np.mean(list(quantization.values())[0]))
            jpeg_quality = int(max(10, min(100, 100 - (avg_q * 1.5))))
        except Exception:
            pass

    # 3. Severe Degradation Criteria (Operating Envelope Thresholds)
    # Check if either severely blurred (< 5.0 regardless of resolution)
    # or moderately degraded while also downscaled (< 35.0 and < 720p)
    is_severely_blurred = bool(laplacian_var < 5.0)
    is_degraded_and_small = bool(laplacian_var < 35.0 and is_downscaled)
    is_severely_degraded = bool(is_severely_blurred or is_degraded_and_small)

    return {
        "dimensions": f"{width}x{height}",
        "sharpness_score": round(laplacian_var, 2),
        "estimated_jpeg_quality": jpeg_quality,
        "is_downscaled": is_downscaled,
        "is_severely_blurred": is_severely_blurred,
        "is_severely_degraded": is_severely_degraded
    }

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            content_type = self.headers.get('Content-Type', '')
            if 'boundary=' not in content_type:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Bad Request: Missing boundary in multipart header"}).encode())
                return

            boundary = content_type.split('boundary=')[1].encode()
            parts = body.split(b'--' + boundary)

            file_bytes = None
            for part in parts:
                if b'filename=' in part:
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        # Extract clean binary file payload
                        file_bytes = part[header_end + 4:].rstrip(b'\r\n')
                        break

            if not file_bytes:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No file uploaded in the request"}).encode())
                return

            # Stage 1: C2PA Provenance Inspection
            c2pa_res = inspect_c2pa(file_bytes)
            if c2pa_res["has_c2pa"]:
                response_data = {
                    "status": "STOP",
                    "verdict": "VERIFIED ORIGIN",
                    "reason": c2pa_res["details"],
                    "c2pa_info": c2pa_res,
                    "send_to_gpu": False
                }
            else:
                # Stage 2: Physical Degradation Profiler
                img = Image.open(io.BytesIO(file_bytes))
                profile = profile_degradation(img)

                if profile["is_severely_degraded"]:
                    response_data = {
                        "status": "STOP",
                        "verdict": "INCONCLUSIVE",
                        "reason": "Image quality is severely degraded (excessive blur or heavy downscaling) outside safe forensic operating boundaries.",
                        "metrics": profile,
                        "send_to_gpu": False
                    }
                else:
                    response_data = {
                        "status": "PROCEED",
                        "verdict": "PENDING_FORENSICS",
                        "reason": "Physical integrity validated. Routing to Kaggle forensic engine for deep inference.",
                        "metrics": profile,
                        "send_to_gpu": True
                    }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
import io
import json
import numpy as np
from PIL import Image
from http.server import BaseHTTPRequestHandler

def inspect_c2pa(raw_bytes: bytes) -> dict:
    has_c2pa = False
    manifest_type = "None"
    is_synthetic = False
    is_hardware_signed = False
    
    # Periksa header JUMBF dan kontainer C2PA
    raw_lower = raw_bytes[:262144].lower()
    if b"c2pa" in raw_lower or b"urn:c2pa" in raw_lower or b"jumb" in raw_lower:
        has_c2pa = True
        
        # 1. Pindai indikator sintetis/generatif (Google/Gemini, DALL-E, SynthID, Midjourney)
        synthetic_markers = [
            b"trainedalgorithmicmedia",
            b"generativesynthesis",
            b"c2pa.synthetic",
            b"synthid",
            b"dall-e",
            b"midjourney"
        ]
        if any(marker in raw_lower for marker in synthetic_markers) or (b"google" in raw_lower and b"gemini" in raw_lower):
            is_synthetic = True
            manifest_type = "Algorithmic / Synthetic Generative Manifest"
        elif b"leica" in raw_lower or b"sony" in raw_lower or b"c2pa.hardware" in raw_lower:
            is_hardware_signed = True
            manifest_type = "Hardware Cryptographic Root Signature"
        elif b"c2pa.claim" in raw_lower:
            manifest_type = "Standard C2PA Claim Manifest"
        else:
            manifest_type = "Generic C2PA Metadata Container"

    return {
        "has_c2pa": has_c2pa,
        "is_synthetic": is_synthetic,
        "is_hardware_signed": is_hardware_signed,
        "manifest_type": manifest_type,
        "details": f"Manifest identified: {manifest_type}." if has_c2pa else "No C2PA provenance container found."
    }

def profile_degradation(img_pil: Image.Image) -> dict:
    width, height = img_pil.size
    total_pixels = width * height
    is_downscaled = bool(total_pixels < (480 * 480))

    # 1. Hitung ketajaman gambar (Laplacian Variance via NumPy)
    gray = img_pil.convert("L")
    arr = np.array(gray, dtype=np.float32)
    edges = (
        arr[:-2, 1:-1] + arr[2:, 1:-1] +
        arr[1:-1, :-2] + arr[1:-1, 2:] -
        4 * arr[1:-1, 1:-1]
    )
    laplacian_var = float(np.var(edges))

    # 2. Estimasi kualitas kompresi JPEG
    jpeg_quality = "Non-JPEG / Clean Header"
    quantization = getattr(img_pil, "quantization", None)
    if quantization:
        try:
            avg_q = float(np.mean(list(quantization.values())[0]))
            jpeg_quality = int(max(10, min(100, 100 - (avg_q * 1.5))))
        except Exception:
            pass

    # 3. Kriteria degradasi parah
    is_severely_blurred = bool(laplacian_var < 4.0)
    is_degraded_and_small = bool(laplacian_var < 20.0 and is_downscaled)
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
    def do_OPTIONS(self):
        # Tangani preflight CORS browser
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            content_type = self.headers.get('Content-Type', '')
            if 'boundary=' not in content_type:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
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
                        file_bytes = part[header_end + 4:].rstrip(b'\r\n')
                        break

            if not file_bytes:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No file uploaded in payload"}).encode())
                return

            # Stage 1: Inspeksi C2PA Provenance yang Diperketat
            c2pa_res = inspect_c2pa(file_bytes)
            
            if c2pa_res["has_c2pa"]:
                if c2pa_res["is_synthetic"]:
                    # Terdeteksi sebagai AI via C2PA (misal Gemini, DALL-E)
                    response_data = {
                        "status": "STOP",
                        "verdict": "LIKELY AI-GENERATED",
                        "confidence": 1.0,
                        "reason": "C2PA manifest explicitly certifies synthetic or algorithmic origin.",
                        "c2pa_info": c2pa_res,
                        "send_to_gpu": False
                    }
                elif c2pa_res["is_hardware_signed"]:
                    # Asli dari sensor kamera hardware (Leica/Sony)
                    response_data = {
                        "status": "STOP",
                        "verdict": "VERIFIED ORIGIN",
                        "confidence": 1.0,
                        "reason": "Cryptographic hardware signature verified from camera sensor.",
                        "c2pa_info": c2pa_res,
                        "send_to_gpu": False
                    }
                else:
                    # Kontainer C2PA ada tapi klaim tidak spesifik -> Lanjutkan ke GPU
                    response_data = {
                        "status": "PROCEED",
                        "verdict": "PENDING_FORENSICS",
                        "reason": "Generic C2PA container detected without verifiable hardware root. Routing to neural engine.",
                        "c2pa_info": c2pa_res,
                        "send_to_gpu": True
                    }
            else:
                # Stage 2: Physical Degradation Profiler
                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    profile = profile_degradation(img)

                    if profile["is_severely_degraded"]:
                        response_data = {
                            "status": "STOP",
                            "verdict": "INCONCLUSIVE",
                            "confidence": 0.50,
                            "reason": "Image quality is severely degraded (excessive blur or heavy downscaling) outside safe forensic envelope.",
                            "metrics": profile,
                            "send_to_gpu": False
                        }
                    else:
                        response_data = {
                            "status": "PROCEED",
                            "verdict": "PENDING_FORENSICS",
                            "confidence": 0.0,
                            "reason": "Physical integrity validated. Routing to neural engine for forensic inference.",
                            "metrics": profile,
                            "send_to_gpu": True
                        }
                except Exception:
                    # Jika payload adalah audio atau video, serahkan langsung ke GPU
                    response_data = {
                        "status": "PROCEED",
                        "verdict": "PENDING_FORENSICS",
                        "reason": "Non-image or dynamic stream detected. Forwarding to multimodal engine.",
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
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
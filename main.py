"""
OPTIX Backend — Zero AI, Zero external APIs
"""

import os
import uuid
import json
import base64
from pathlib import Path

import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

BASE    = Path(os.environ.get("DATA_DIR", "./data"))
UPLOADS = BASE / "uploads"
RESULTS = BASE / "results"
UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

jobs: dict[str, str] = {}

app = FastAPI(title="OPTIX")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Location", "Upload-Offset", "Upload-Length",
        "Tus-Resumable", "Tus-Version", "Tus-Max-Size", "Tus-Extension",
    ],
)

TUS_VER = "1.0.0"

def _h(extra: dict = {}) -> dict:
    return {"Tus-Resumable": TUS_VER, "Tus-Version": TUS_VER, **extra}


@app.options("/upload")
async def tus_options():
    return Response(status_code=204, headers=_h({
        "Tus-Max-Size":  str(50 * 1024 ** 3),
        "Tus-Extension": "creation,termination",
    }))


@app.post("/upload")
async def tus_create(request: Request):
    length_str = request.headers.get("Upload-Length", "")
    if not length_str.isdigit():
        return Response(status_code=400, content="Upload-Length required")

    meta: dict[str, str] = {}
    for pair in request.headers.get("Upload-Metadata", "").split(","):
        pair = pair.strip()
        if " " in pair:
            k, v = pair.split(" ", 1)
            try:
                meta[k] = base64.b64decode(v).decode()
            except Exception:
                meta[k] = v

    uid = str(uuid.uuid4())
    jobs[uid] = "uploading"
    info = {"id": uid, "length": int(length_str), "offset": 0, "meta": meta}
    (UPLOADS / f"{uid}.json").write_text(json.dumps(info))
    (UPLOADS / f"{uid}.bin").write_bytes(b"")
    return Response(status_code=201, headers=_h({"Location": f"/upload/{uid}"}))


@app.head("/upload/{uid}")
async def tus_head(uid: str):
    p = UPLOADS / f"{uid}.json"
    if not p.exists():
        return Response(status_code=404)
    info = json.loads(p.read_text())
    return Response(status_code=200, headers=_h({
        "Upload-Offset": str(info["offset"]),
        "Upload-Length": str(info["length"]),
        "Cache-Control": "no-store",
    }))


@app.patch("/upload/{uid}")
async def tus_patch(uid: str, request: Request, bg: BackgroundTasks):
    p = UPLOADS / f"{uid}.json"
    if not p.exists():
        return Response(status_code=404)
    info = json.loads(p.read_text())
    client_offset = int(request.headers.get("Upload-Offset", -1))
    if client_offset != info["offset"]:
        return Response(status_code=409, content="Offset mismatch")
    body = await request.body()
    async with aiofiles.open(UPLOADS / f"{uid}.bin", "ab") as f:
        await f.write(body)
    new_offset = info["offset"] + len(body)
    info["offset"] = new_offset
    p.write_text(json.dumps(info))
    if new_offset >= info["length"]:
        jobs[uid] = "queued"
        bg.add_task(_analyse, uid, str(UPLOADS / f"{uid}.bin"))
    return Response(status_code=204, headers=_h({"Upload-Offset": str(new_offset)}))


@app.delete("/upload/{uid}")
async def tus_delete(uid: str):
    for ext in [".bin", ".json"]:
        fp = UPLOADS / f"{uid}{ext}"
        if fp.exists():
            fp.unlink()
    jobs.pop(uid, None)
    return Response(status_code=204, headers=_h())


@app.get("/status/{uid}")
async def status(uid: str):
    return JSONResponse({"id": uid, "status": jobs.get(uid, "unknown")})


@app.get("/results/{uid}")
async def results(uid: str):
    f = RESULTS / uid / "analysis.json"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return JSONResponse(json.loads(f.read_text()))


@app.get("/results/{uid}/heatmap.png")
async def heatmap_image(uid: str):
    f = RESULTS / uid / "heatmap.png"
    if not f.exists():
        return Response(status_code=404)
    return FileResponse(str(f), media_type="image/png")


@app.get("/results/{uid}/overlay.png")
async def overlay_image(uid: str):
    f = RESULTS / uid / "overlay.png"
    if not f.exists():
        return Response(status_code=404)
    return FileResponse(str(f), media_type="image/png")


def _analyse(uid: str, video_path: str) -> None:
    """
    Maximum accuracy heatmap:
    - MOG2 with varThreshold=16 (very sensitive — catches every movement)
    - Every 2nd frame sampled (ultra accurate even for 6h videos)
    - Gaussian blur for smooth result
    - JET colormap: blue=nothing, green=some, yellow=busy, red=HOTTEST
    - Pure heatmap + overlay on actual video frame
    """
    try:
        jobs[uid] = "processing"
        out = RESULTS / uid
        out.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Empty video")

        h, w = frame0.shape[:2]
        accum = np.zeros((h, w), dtype=np.float32)

        # Very sensitive background subtractor
        # varThreshold=16 = catches slow walkers, subtle movements
        # history=200 = adapts to lighting changes quickly
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=16,
            detectShadows=False
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total = 0
        sampled = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total += 1

            # Every 2nd frame = maximum accuracy
            # For 6h video at 25fps = ~540,000 frames → 270,000 processed
            if total % 2 != 0:
                continue

            fg = mog2.apply(frame)

            # Low threshold = catch every movement including slow ones
            _, thresh = cv2.threshold(fg, 50, 1, cv2.THRESH_BINARY)
            accum += thresh.astype(np.float32)
            sampled += 1

        cap.release()

        if sampled == 0:
            raise RuntimeError("No frames processed")

        # Smooth the heatmap — makes hot zones clear and beautiful
        accum_blur = cv2.GaussianBlur(accum, (31, 31), 0)

        # Normalize 0-255
        norm = cv2.normalize(accum_blur, None, 0, 255, cv2.NORM_MINMAX)
        norm_u8 = norm.astype(np.uint8)

        # JET colormap: blue → green → yellow → red
        heatmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
        cv2.imwrite(str(out / "heatmap.png"), heatmap)

        # Overlay heatmap on a real frame from the middle of the video
        cap2 = cv2.VideoCapture(video_path)
        n = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ret2, mid = cap2.read()
        cap2.release()

        if ret2:
            overlay = cv2.addWeighted(mid, 0.45, heatmap, 0.65, 0)
            cv2.imwrite(str(out / "overlay.png"), overlay)

        # Zone stats (10x10 grid) for the legend
        ROWS, COLS = 10, 10
        zh, zw = h // ROWS, w // COLS
        zones = []
        for r in range(ROWS):
            for c in range(COLS):
                y1, y2 = r * zh, (r + 1) * zh
                x1, x2 = c * zw, (c + 1) * zw
                score = float(accum_blur[y1:y2, x1:x2].mean())
                zones.append({
                    "id": f"R{r+1}C{c+1}",
                    "score": round(score, 2),
                    "x1": round(x1 / w * 100, 1),
                    "y1": round(y1 / h * 100, 1),
                    "x2": round(x2 / w * 100, 1),
                    "y2": round(y2 / h * 100, 1),
                })
        zones.sort(key=lambda z: z["score"], reverse=True)
        max_s = zones[0]["score"] if zones else 1
        for z in zones:
            z["intensity"] = round(z["score"] / max_s * 100)

        result = {
            "id": uid,
            "status": "done",
            "frames_total": total,
            "frames_sampled": sampled,
            "resolution": f"{w}x{h}",
            "heatmap_url": f"/results/{uid}/heatmap.png",
            "overlay_url": f"/results/{uid}/overlay.png",
            "zones": zones,
        }

        (out / "analysis.json").write_text(json.dumps(result, indent=2))
        jobs[uid] = "done"

    except Exception as e:
        jobs[uid] = f"error: {e}"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import subprocess
import os

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    image_path: str = "examples/03/image.jpg"
    action_path: str = "examples/03"
    frame_num: int = 81

@app.post("/generate")
def generate_video(req: PromptRequest):
    command = [
        "torchrun", "--nproc_per_node=1", "generate.py",
        "--task", "i2v-A14B",
        "--size", "480*832",
        "--ckpt_dir", "lingbot-world-v2-14b-causal-fast",
        "--image", req.image_path,
        "--action_path", req.action_path,
        "--prompt", req.prompt,
        "--frame_num", str(req.frame_num)
    ]
    
    try:
        process = subprocess.run(command, capture_output=True, text=True, cwd="/workspace/lingbot-world-v2")
        if process.returncode == 0:
            return {"status": "success", "message": "Generation completed.", "output": process.stdout}
        else:
            return {"status": "error", "message": "Generation failed.", "error": process.stderr, "out": process.stdout}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/download/{filename:path}")
def download_file(filename: str):
    file_path = os.path.join("/workspace/lingbot-world-v2", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/files")
def list_files():
    try:
        files = []
        for root, dirs, filenames in os.walk("/workspace/lingbot-world-v2"):
            for f in filenames:
                if f.endswith(".mp4"):
                    rel_dir = os.path.relpath(root, "/workspace/lingbot-world-v2")
                    if rel_dir == ".":
                        files.append(f)
                    else:
                        files.append(os.path.join(rel_dir, f))
        return {"videos": files}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)

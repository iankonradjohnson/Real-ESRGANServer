from flask import Flask, request, jsonify, send_file
import uuid
import os
import shutil
import zipfile
from threading import Thread
import subprocess

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

WORKSPACE_DIR = "workspace"
BASE_DIR = os.path.join(WORKSPACE_DIR, "data")
IN_DIR = os.path.join(BASE_DIR, "in")
OUT_DIR = os.path.join(BASE_DIR, "out")
TMP_DIR = "/tmp/runpod_jobs"
ESRGAN_SCRIPT = os.path.join(WORKSPACE_DIR, "Real-ESRGAN/inference_realesrgan.py")
JOBS = {}  # job_id -> status/info

os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

def find_latest_zip(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    files = sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, files[0]) if files else None

def unzip_to_dir(zip_path, target_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def zip_dir(source_dir, output_zip):
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', source_dir)

def process_job(job_id):
    try:
        JOBS[job_id]["status"] = "locating_zip"
        zip_path = find_latest_zip("/workspace")
        if not zip_path:
            raise FileNotFoundError("No .zip file found in /workspace")

        JOBS[job_id]["status"] = "unzipping"
        shutil.rmtree(IN_DIR, ignore_errors=True)
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        os.makedirs(IN_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        unzip_to_dir(zip_path, IN_DIR)

        JOBS[job_id]["status"] = "processing"
        subprocess.run([
            "python",
            ESRGAN_SCRIPT,
            "-i", IN_DIR,
            "-o", OUT_DIR,
            "-n", "net_g_1000000",
            "-t", "1000",
            "--tile_pad", "0",
            "-p", ".75"
        ], check=True)

        JOBS[job_id]["status"] = "zipping"
        output_zip_path = os.path.join(TMP_DIR, f"{job_id}_out.zip")
        zip_dir(OUT_DIR, output_zip_path)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["output_path"] = output_zip_path

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

@app.route("/jobs", methods=["POST"])
def create_job():
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending"}

    thread = Thread(target=process_job, args=(job_id,))
    thread.start()

    return jsonify({"job_id": job_id})

@app.route("/jobs/<job_id>/status", methods=["GET"])
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error")
    })

@app.route("/jobs/<job_id>/download", methods=["GET"])
def download_output(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "completed" or "output_path" not in job:
        return jsonify({"error": "Output not ready"}), 400

    return send_file(job["output_path"], as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

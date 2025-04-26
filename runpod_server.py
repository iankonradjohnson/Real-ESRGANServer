from flask import Flask, request, jsonify
import uuid
import os
import shutil
import zipfile
from threading import Thread
import subprocess
from google.cloud import storage

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

WORKSPACE_DIR = "/workspace"
BASE_DIR = os.path.join(WORKSPACE_DIR, "data")
IN_DIR = os.path.join(BASE_DIR, "in")
OUT_DIR = os.path.join(BASE_DIR, "out")
TMP_DIR = "/tmp/runpod_jobs"
REAL_ESRGAN_DIR = os.path.join(WORKSPACE_DIR, "Real-ESRGAN")
JOBS = {}  # job_id -> status/info
GCS_BUCKET = os.environ.get("GCS_BUCKET")  # Bucket name from env var

os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

def find_latest_zip(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    files = sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, files[0]) if files else None

def unzip_and_get_input_dir(zip_path, extract_root):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_root)
    folder_name = os.path.splitext(os.path.basename(zip_path))[0]
    return os.path.join(extract_root, folder_name)

def zip_dir(source_dir, output_zip):
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', source_dir)

def upload_to_gcs(source_path, dest_blob_name):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(source_path)
    return blob.public_url

def process_job(job_id):
    try:
        JOBS[job_id]["status"] = "receiving"
        receive_code = JOBS[job_id]["receive_code"]
        model_name = JOBS[job_id]["model_name"]

        subprocess.run(f"runpodctl receive {receive_code}", shell=True, check=True)

        JOBS[job_id]["status"] = "unzipping"
        zip_path = find_latest_zip(WORKSPACE_DIR)
        if not zip_path:
            raise FileNotFoundError("No .zip file found after receive")

        shutil.rmtree(IN_DIR, ignore_errors=True)
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        os.makedirs(IN_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)

        actual_input_dir = unzip_and_get_input_dir(zip_path, IN_DIR)

        JOBS[job_id]["status"] = "processing"
        with subprocess.Popen([
            "python", "inference_realesrgan.py",
            "-i", actual_input_dir,
            "-o", OUT_DIR,
            "-n", model_name,
            "-t", "1000",
            "--tile_pad", "0"
        ], cwd=REAL_ESRGAN_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='')
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)

        JOBS[job_id]["status"] = "zipping"
        output_zip_path = os.path.join(TMP_DIR, f"{job_id}_out.zip")
        zip_dir(OUT_DIR, output_zip_path)

        JOBS[job_id]["status"] = "uploading"
        public_url = upload_to_gcs(output_zip_path, f"jobs/{job_id}_out.zip")

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["gcs_url"] = public_url

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

@app.route("/jobs", methods=["POST"])
def create_job():
    data = request.get_json()
    if not data or "receive_code" not in data or "model_name" not in data:
        return jsonify({"error": "Missing receive_code or model_name in request body"}), 400

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "pending",
        "receive_code": data["receive_code"],
        "model_name": data["model_name"]
    }

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
        "error": job.get("error"),
        "gcs_url": job.get("gcs_url")
    })

@app.route("/jobs/<job_id>/download_url", methods=["GET"])
def get_download_url(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "completed" or "gcs_url" not in job:
        return jsonify({"error": "Output not ready"}), 400

    return jsonify({"download_url": job["gcs_url"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

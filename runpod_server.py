from flask import Flask, request, jsonify, send_file
import uuid
import os
import shutil
import zipfile
from threading import Thread
import subprocess

app = Flask(__name__)
# WORKSPACE_DIR = "/Users/iankonradjohnson/workspace"
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


def unzip_to_dir(zip_path, target_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)


def zip_dir(source_dir, output_zip):
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', source_dir)


def process_job(job_id, zip_path):
    try:
        JOBS[job_id]["status"] = "unzipping"
        # Clear input/output dirs
        shutil.rmtree(IN_DIR, ignore_errors=True)
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        os.makedirs(IN_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)

        unzip_to_dir(zip_path, IN_DIR)

        JOBS[job_id]["status"] = "processing"
        # Run Real-ESRGAN using the Python script
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
    if 'file' not in request.files:
        return jsonify({"error": "Missing file upload"}), 400

    file = request.files['file']
    job_id = str(uuid.uuid4())
    input_zip_path = os.path.join(TMP_DIR, f"{job_id}.zip")
    file.save(input_zip_path)

    JOBS[job_id] = {"status": "pending", "input_path": input_zip_path}

    thread = Thread(target=process_job, args=(job_id, input_zip_path))
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

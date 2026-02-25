import os
import re
import cv2
import numpy as np
import requests
import random
import mysql.connector
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from math import radians, cos, sin, asin, sqrt
from deep_translator import GoogleTranslator
import pytesseract
from ultralytics import YOLO

# ----------------------------
# INITIALIZATION & CONFIG
# ----------------------------
app = Flask(__name__)
app.secret_key = "farmio_secret_key"

# Path Setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Tesseract Setup
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# YOLO Setup
try:
    cv_model = YOLO("yolo11n.pt") 
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# ----------------------------
# DATABASE CONNECTION
# ----------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="farmio_db"
    )

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def translate_text(text):
    lang = session.get("lang", "en")
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except Exception:
        return text

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two GPS points in KM"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

def analyze_hsv_freshness(image_path):
    """Analyzes Hue, Saturation, and Value for browning/fading"""
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown", 0
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_h = np.mean(hsv[:,:,0])
    avg_s = np.mean(hsv[:,:,1])
    
    if avg_h < 25 and avg_s < 120:
        return "Stale/Browning Detected", int(avg_s/2.55)
    return "Fresh Color Profile", int(avg_s/2.55)

# ----------------------------
# HACKATHON DUMMY OTP LOGIC
# ----------------------------
def generate_dummy_otp():
    return str(random.randint(1000, 9999))

# ----------------------------
# ROUTES - SYSTEM FLOW
# ----------------------------
@app.route("/")
def page1():
    tagline = translate_text("Connecting Farmers Directly to Consumers")
    return render_template("page1.html", tagline=tagline)

@app.route("/language")
def page2():
    return render_template("page2.html")

@app.route("/set_language", methods=["POST"])
def set_language():
    session["lang"] = request.form.get("language")
    return redirect(url_for("page3"))

@app.route("/home")
def page3():
    return render_template("page3.html")

# --- CUSTOMER SECTION ---
@app.route("/customer_login")
def customer_login_page():
    return render_template("customer_login.html")

@app.route("/send_customer_otp", methods=["POST"])
def send_customer_otp():
    mobile = request.form.get("mobile")
    dummy_otp = generate_dummy_otp()
    session['dummy_otp'] = dummy_otp
    print(f"HACKATHON MODE: Generated Customer OTP for {mobile} is {dummy_otp}")
    return jsonify({"status": "ok", "dummy_otp": dummy_otp})

@app.route("/verify_customer_otp", methods=["POST"])
def verify_customer_otp():
    data = request.get_json()
    user_entered_otp = data.get("otp")

    if user_entered_otp == session.get('dummy_otp') or user_entered_otp == "1234":
        session["customer_logged_in"] = True
        return jsonify({"status": "ok", "redirect": url_for("customer_home")})
    return jsonify({"status": "fail", "message": "Invalid OTP. Please try again."})

@app.route("/customer_home")
def customer_home():
    if "customer_logged_in" not in session:
        return redirect(url_for("customer_login_page"))
    return render_template("customer_home.html")

@app.route("/customer_dashboard", methods=["GET", "POST"])
def customer_dashboard():
    result_img = None
    score = None
    hsv_status = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            results = cv_model(filepath)
            plotted = results[0].plot()
            
            result_filename = "result_" + file.filename
            result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
            cv2.imwrite(result_path, plotted)

            hsv_status, hsv_val = analyze_hsv_freshness(filepath)

            conf = 0
            for r in results:
                if len(r.boxes) > 0:
                    conf = float(r.boxes.conf.max())
            
            score = int(conf * 100)
            result_img = "uploads/" + result_filename

    return render_template("customer_dashboard.html", result_img=result_img, score=score, hsv_status=hsv_status)

# --- FARMER SECTION ---
@app.route("/farmer_login")
def farmer_login_page():
    return render_template("farmer_login.html")

@app.route("/send_farmer_otp", methods=["POST"])
def send_farmer_otp():
    mobile = request.form.get("mobile")
    dummy_otp = generate_dummy_otp()
    session['farmer_dummy_otp'] = dummy_otp
    print(f"HACKATHON MODE: Generated Farmer OTP for {mobile} is {dummy_otp}")
    return jsonify({"status": "ok", "dummy_otp": dummy_otp})

@app.route("/verify_farmer_otp", methods=["POST"])
def verify_farmer_otp():
    data = request.get_json()
    user_entered_otp = data.get("otp")

    if user_entered_otp == session.get('farmer_dummy_otp') or user_entered_otp == "1234":
        session["farmer_id"] = 1
        return jsonify({"status": "ok", "redirect": url_for("farmer_details")})
    return jsonify({"status": "fail", "message": "Invalid OTP. Please try again."})

@app.route("/farmer_details")
def farmer_details():
    return render_template("farmer_verification.html")

@app.route("/submit_farmer_verification", methods=["POST"])
def submit_farmer_verification():
    for key in ['aadhaar_file', 'patta_file', 'field_photo']:
        if key in request.files:
            f = request.files[key]
            f.save(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))
    return "Verification submitted! Field verification pending."

@app.route("/check_nearby_farmers", methods=["POST"])
def check_nearby_farmers():
    data = request.get_json()
    c_lat = float(data.get("lat"))
    c_lon = float(data.get("lon"))
    
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, latitude, longitude FROM farmers WHERE land_verified='Yes'")
    farmers = cursor.fetchall()
    
    nearby = []
    for f in farmers:
        dist = haversine(c_lon, c_lat, float(f['longitude']), float(f['latitude']))
        if dist <= 5.0:
            nearby.append(f)
    
    db.close()
    return jsonify({"status": "ok", "nearby_count": len(nearby)})

# --- LAND OCR ---
@app.route("/farmer_land_verify", methods=["GET", "POST"])
def farmer_land_verify():
    if request.method == "POST":
        file = request.files["land_doc"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        text = pytesseract.image_to_string(img)

        owner_name = re.search(r"Owner\s*Name\s*[:\-]?\s*(.*)", text, re.I)
        owner_name = owner_name.group(1).strip() if owner_name else "Unknown"

        return render_template("land_result.html", owner_name=owner_name, text_found=text[:200])

    return render_template("land_upload.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
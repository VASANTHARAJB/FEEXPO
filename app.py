import os
import re
import cv2
import numpy as np
import requests
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

# Path Setup - Using relative paths for better portability
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Tesseract & YOLO Setup
# Note: Ensure Tesseract is installed at this path on the laptop
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Use the model file found in your project root
try:
    cv_model = YOLO("yolo11n.pt") 
except Exception as e:
    print(f"Error loading YOLO model: {e}")

API_KEY = "215a66be-fcc4-11f0-a6b2-0200cd936042"

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
    """Step 6 & 7: Analyzes Hue, Saturation, and Value for browning/fading"""
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown", 0
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_h = np.mean(hsv[:,:,0]) # Hue
    avg_s = np.mean(hsv[:,:,1]) # Saturation
    
    # Simple threshold: Low hue/saturation often indicates browning or decay
    if avg_h < 25 and avg_s < 120:
        return "Stale/Browning Detected", int(avg_s/2.55)
    return "Fresh Color Profile", int(avg_s/2.55)

# ----------------------------
# AUTHENTICATION (2FACTOR)
# ----------------------------

def send_otp_2factor(mobile):
    url = f"https://2factor.in/API/V1/{API_KEY}/SMS/{mobile}/AUTOGEN"
    try:
        response = requests.get(url)
        data = response.json()
        if data['Status'] == 'Success':
            session['session_id'] = data['Details']
            return True
        return False
    except Exception:
        return False

def verify_otp_2factor(otp):
    session_id = session.get('session_id')
    if not session_id:
        return False, "Session expired."
    
    url = f"https://2factor.in/API/V1/{API_KEY}/SMS/VERIFY/{session_id}/{otp}"
    try:
        response = requests.get(url).json()
        if response['Status'] == 'Success':
            return True, "Verified"
        return False, "Invalid OTP"
    except Exception:
        return False, "Error"

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

@app.route("/verify_customer_otp", methods=["POST"])
def verify_customer_otp():
    data = request.get_json()
    success, msg = verify_otp_2factor(data.get("otp"))
    if success:
        session["customer_logged_in"] = True
        return jsonify({"status": "ok", "redirect": url_for("customer_home")})
    return jsonify({"status": "fail", "message": msg})

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

            # 1. YOLO Detection (Step 3)
            results = cv_model(filepath)
            plotted = results[0].plot()
            
            result_filename = "result_" + file.filename
            result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
            cv2.imwrite(result_path, plotted)

            # 2. HSV Analysis (Step 6 & 7)
            hsv_status, hsv_val = analyze_hsv_freshness(filepath)

            # 3. Decision Engine (Step 9)
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

@app.route("/submit_farmer_verification", methods=["POST"])
def submit_farmer_verification():
    # Save files and data
    for key in ['aadhaar_file', 'patta_file', 'field_photo']:
        f = request.files[key]
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))
    
    # Store coordinates for the Geofencing feature
    # In a real app, save request.form['latitude'] and longitude to DB here.
    return "Verification submitted! Field verification pending."

@app.route("/check_nearby_farmers", methods=["POST"])
def check_nearby_farmers():
    """Consumer Distance Check API (Step: Farmer Location Store)"""
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
        if dist <= 5.0: # 5km Radius Green Signal
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

        # Extraction logic
        owner_name = re.search(r"Owner\s*Name\s*[:\-]?\s*(.*)", text, re.I)
        owner_name = owner_name.group(1).strip() if owner_name else "Unknown"

        return render_template("land_result.html", owner_name=owner_name, text_found=text[:200])

    return render_template("land_upload.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
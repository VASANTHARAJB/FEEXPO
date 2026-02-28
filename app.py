import os
import re
import cv2
import numpy as np
import requests
import random
import mysql.connector
import json
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from math import radians, cos, sin, asin, sqrt
import pytesseract
from google import genai
from ultralytics import YOLO

# ----------------------------
# INITIALIZATION & CONFIG
# ----------------------------
app = Flask(__name__)
app.secret_key = "farmio_secret_key"

# Path Setup
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Google AI Studio (Gemini) Setup - NEW SDK
# PASTE YOUR ACTUAL KEY HERE
gemini_client = genai.Client(api_key="YOUR_GOOGLE_AI_STUDIO_KEY")

# Tesseract Setup (Keep commented out if Tesseract isn't installed on the presentation laptop)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
# GLOBAL JSON TRANSLATOR (LIGHTNING FAST)
# ----------------------------
# Load the JSON file into memory when the server starts
try:
    with open('translations.json', 'r', encoding='utf-8') as f:
        translations = json.load(f)
except Exception as e:
    print(f"Warning: Could not load translations.json: {e}")
    translations = {}

@app.context_processor
def inject_translator():
    def t(text):
        lang = session.get("lang", "en")
        
        # If English or empty, return immediately (Speed Boost)
        if lang == "en" or not text:
            return text
            
        # Check if the language and exact text exist in our JSON file
        if lang in translations and text in translations[lang]:
            return translations[lang][text]
            
        # Fallback: Return English if translation is missing from JSON
        return text 
            
    return dict(t=t)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two GPS points in KM using Haversine formula"""
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

def generate_dummy_otp():
    return str(random.randint(1000, 9999))

# ----------------------------
# ROUTES - SYSTEM FLOW
# ----------------------------
@app.route("/")
def page1():
    return render_template("page1.html")

@app.route("/language")
def page2():
    return render_template("page2.html")

@app.route("/set_language", methods=["POST"])
def set_language():
    selected_lang = request.form.get("language")
    session["lang"] = selected_lang
    print(f"✅ User switched language to: {selected_lang}")
    
    # FIX: Send them to the next page (page3) instead of refreshing!
    return redirect(url_for('page3'))

@app.route("/home")
def page3():
    return render_template("page3.html")

# --- AI ADVICE (Using Gemini SDK) ---
@app.route("/get_ai_advice", methods=["POST"])
def get_ai_advice():
    veg_name = request.json.get("vegetable")
    prompt = f"I just bought fresh {veg_name} from a local farmer. Give me a 2-sentence healthy cooking tip for it."
    try:
        response = gemini_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return jsonify({"advice": response.text})
    except Exception:
        return jsonify({"advice": "Wash thoroughly and enjoy your farm-fresh meal!"})

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
        return jsonify({"status": "ok", "redirect": url_for("customer_dashboard")})
    return jsonify({"status": "fail", "message": "Invalid OTP. Please try again."})

@app.route("/customer_dashboard", methods=["GET", "POST"])
def customer_dashboard():
    result_img = None
    score = None
    hsv_status = None
    label = "Unknown"

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename: 
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # YOLO Logic
            results = cv_model(filepath)
            plotted = results[0].plot()
            
            result_filename = "result_" + file.filename
            result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
            cv2.imwrite(result_path, plotted)

            # HSV Color Logic
            hsv_status, hsv_val = analyze_hsv_freshness(filepath)

            conf = 0
            for r in results:
                if len(r.boxes) > 0:
                    conf = float(r.boxes.conf.max())
                    class_id = int(r.boxes.cls[0])
                    label = r.names[class_id]
            
            score = int(conf * 100)
            result_img = "static/uploads/" + result_filename

    return render_template("customer_dashboard.html", result_img=result_img, score=score, hsv_status=hsv_status, label=label)

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
    # 1. Capture GPS
    lat = request.form.get("latitude")
    lon = request.form.get("longitude")
    
    # 2. Capture Legal Details
    aadhaar_num = request.form.get("aadhaar_number")
    patta_num = request.form.get("patta_number")
    land_type = request.form.get("land_type")
    
    print(f"✅ RECEIVED KYC: Aadhaar: {aadhaar_num}, Patta: {patta_num}, Type: {land_type}")
    
    # 3. Save files
    for key in ['aadhaar_file', 'patta_file', 'field_photo']:
        if key in request.files:
            f = request.files[key]
            if f.filename:
                f.save(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))

    # 4. Reverse Geocoding
    address_name = "Location Verified"
    try:
        geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {'User-Agent': 'FarmioApp/1.0'}
        response = requests.get(geo_url, headers=headers).json()
        address_name = response.get('display_name', 'Location Verified')
        address_name = ", ".join(address_name.split(",")[:3]) 
    except Exception:
        address_name = "Chennai, Tamil Nadu"

    # 5. Save to session
    session['dummy_farmer_lat'] = lat
    session['dummy_farmer_lon'] = lon
    session['farmer_address'] = address_name

    return f"""
    <html>
        <body style="font-family: 'Poppins', sans-serif; text-align: center; background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); padding-top: 100px; margin: 0; min-height: 100vh;">
            <div style="background: white; padding: 50px; border-radius: 24px; display: inline-block; box-shadow: 0 20px 50px rgba(0,0,0,0.05); max-width: 600px; width: 100%;">
                <h1 style="color: #28a745; margin-top: 0; font-size: 32px;">✅ KYC Successful!</h1>
                <p style="font-size: 18px; color: #4a5568;">Your documents and Live Location are locked.</p>
                <div style="background: #f0fff4; color: #22543d; padding: 20px; border-radius: 12px; margin: 25px 0; border: 1px solid #c6f6d5; text-align: left;">
                    <strong>📍 Registered Address:</strong><br>{address_name}<br><br>
                    <strong>📝 Land Type:</strong> {land_type}
                </div>
                <a href="/customer_dashboard" style="display: inline-block; background: linear-gradient(45deg, #11998e, #38ef7d); color: white; padding: 16px 30px; text-decoration: none; border-radius: 12px; font-weight: bold; font-size: 18px; margin-top: 10px; box-shadow: 0 10px 20px rgba(56, 239, 125, 0.3);">
                    Test Customer Dashboard 🛒
                </a>
            </div>
        </body>
    </html>
    """

@app.route("/check_nearby_farmers", methods=["POST"])
def check_nearby_farmers():
    data = request.get_json()
    c_lat = float(data.get("lat"))
    c_lon = float(data.get("lon"))
    
    f_lat = session.get('dummy_farmer_lat')
    f_lon = session.get('dummy_farmer_lon')
    f_addr = session.get('farmer_address', 'Nearby Farmer')
    
    nearby_count = 0
    dist = 0
    if f_lat and f_lon:
        dist = haversine(c_lon, c_lat, float(f_lon), float(f_lat))
        
        # 10.0 km radius limit
        if dist <= 10.0:
            nearby_count = 1
            
    return jsonify({
        "status": "ok", 
        "nearby_count": nearby_count,
        "distance": round(dist, 2) if f_lat else 0,
        "location_name": f_addr,
        "farmer_lat": f_lat,
        "farmer_lon": f_lon
    })

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
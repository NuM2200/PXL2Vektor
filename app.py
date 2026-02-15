import streamlit as st
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io
import os
import subprocess
import tempfile
import base64
from streamlit_cropper import st_cropper
import csv
from datetime import datetime
import pandas as pd

# --- 1. KONFIGURATION ---
st.set_page_config(page_title="Plotter Converter", page_icon="✂️", layout="wide")

# --- 2. LOGGER FUNKTION (Speichert Daten in Excel/CSV) ---
LOG_FILE = "nutzungsdaten.csv"

def log_event(event_type, detail):
    """Schreibt eine Zeile in die CSV Datei"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prüfen ob Datei existiert, wenn nicht: Header schreiben
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Zeitstempel", "Event", "Detail"]) # Kopfzeile
        
        writer.writerow([timestamp, event_type, detail])

# --- 3. HELFER FUNKTIONEN (Bildverarbeitung) ---

def bild_vorbehandeln(image, helligkeit, kontrast):
    adjusted = cv2.convertScaleAbs(image, alpha=kontrast, beta=helligkeit)
    return adjusted

def bild_verbessern_clahe(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def bild_laden_und_freistellen(pil_image, hintergrund_entfernen):
    try:
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        input_data = img_byte_arr.getvalue()
        if hintergrund_entfernen:
            output_data = remove(input_data)
            pil_image = Image.open(io.BytesIO(output_data)).convert("RGBA")
            white_bg = Image.new("RGBA", pil_image.size, "WHITE")
            white_bg.paste(pil_image, (0, 0), pil_image)
            final_image = white_bg.convert("RGB")
        else:
            final_image = pil_image.convert("RGB")
        return np.array(final_image)[:, :, ::-1].copy()
    except Exception: return None

def erstelle_skizze_live(image, algorithmus, blur, thresh, noise, erode, invert, light_fix):
    if light_fix: image = bild_verbessern_clahe(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if algorithmus == "Künstlerisch (Skizze)":
        inv = 255 - gray
        if blur % 2 == 0: blur += 1
        blurred = cv2.GaussianBlur(inv, (blur, blur), 0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        _, binary = cv2.threshold(sketch, thresh, 255, cv2.THRESH_BINARY)
    else: 
        block_size = blur 
        if block_size % 2 == 0: block_size += 1
        if block_size < 3: block_size = 3
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, (thresh - 128) / 10)

    if erode > 0:
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=erode)
    
    inv_bin = cv2.bitwise_not(binary) 
    contours, _ = cv2.findContours(inv_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sauber = np.zeros_like(inv_bin)
    for cnt in contours:
        if cv2.contourArea(cnt) > noise: cv2.drawContours(sauber, [cnt], -1, 255, -1)
    
    final_result = cv2.bitwise_not(sauber)
    if invert: return cv2.bitwise_not(final_result)
    return final_result

def erstelle_svg_datei(binary_image):
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp_bmp:
        cv2.imwrite(tmp_bmp.name, binary_image)
        bmp_path = tmp_bmp.name
    svg_path = bmp_path.replace(".bmp", ".svg")
    try:
        subprocess.run(["potrace", bmp_path, "-s", "--turdsize", "10", "--alphamax", "1.0", "-o", svg_path], stdout=subprocess.DEVNULL, check=True)
        with open(svg_path, "r", encoding="utf-8") as f: svg_content = f.read()
        os.remove(bmp_path); os.remove(svg_path)
        return svg_content
    except Exception: return None

def render_svg_html(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" style="max-width: 100%; border: 1px solid #ddd; padding: 10px;"/>'


# --- 4. HAUPTPROGRAMM ---

# Tracking: Seitenaufruf loggen (passiert bei jedem Reload)
if 'has_logged_start' not in st.session_state:
    log_event("Besuch", "Seite geöffnet / Neu geladen")
    st.session_state['has_logged_start'] = True

# --- UI LOGIK (FREEMIUM) ---
st.sidebar.title("💎 Version wählen")
app_mode = st.sidebar.radio("Status:", ["Free Version", "Pro Version (Upgrade)"])

# Tracking: Wenn der Nutzer den Modus wechselt
if 'last_mode' not in st.session_state or st.session_state['last_mode'] != app_mode:
    log_event("Modus Wechsel", f"Gewechselt zu: {app_mode}")
    st.session_state['last_mode'] = app_mode

is_pro = (app_mode == "Pro Version (Upgrade)")

st.title("✂️ Plotter Converter")
if not is_pro:
    st.info("Du nutzt die **Free Version**. Upgrade auf **Pro**, um Zuschneiden, Lichtkorrektur und Profi-Filter freizuschalten!")
else:
    st.success("✨ **Pro Version** aktiv. Alle Features freigeschaltet!")

# --- ADMIN BEREICH (Versteckt in Sidebar ganz unten) ---
# Hier kannst du die Daten herunterladen!
st.sidebar.markdown("---")
with st.sidebar.expander("🔐 Admin / Auswertung"):
    password = st.text_input("Passwort", type="password")
    if password == "study2024": # <--- DEIN PASSWORT
        if os.path.exists(LOG_FILE):
            st.write("📊 Nutzungsdaten:")
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df)
            
            with open(LOG_FILE, "rb") as f:
                st.download_button("📥 CSV Datei herunterladen", f, file_name="nutzungsdaten.csv")
        else:
            st.warning("Noch keine Daten vorhanden.")

# --- STEUERUNG ---
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Einstellungen")

if is_pro:
    with st.sidebar.expander("🛠️ Vorbehandlung (Pro)", expanded=True):
        pre_contrast = st.slider("Kontrast", 0.5, 3.0, 1.0, 0.1)
        pre_brightness = st.slider("Helligkeit", -100, 100, 0, 5)
    
    algo = st.sidebar.radio("Stil:", ["Künstlerisch (Skizze)", "Scanner (Adaptiv)"])
    p_blur = st.sidebar.slider("Detailgrad", 1, 151, 55, 2)
    p_thresh = st.sidebar.slider("Schwelle", 0, 255, 235, 1)
    p_noise = st.sidebar.slider("Rauschen entfernen", 0, 200, 50, 1)
    p_erode = st.sidebar.slider("Linien dicker", 0, 3, 0, 1)
    
    st.sidebar.subheader("Optionen")
    opt_light = st.sidebar.checkbox("💡 Autom. Licht-Balance", value=True)
    opt_bg_weg = st.sidebar.checkbox("Hintergrund entfernen", value=True)
    opt_invert = st.sidebar.checkbox("Invertieren", value=False)
else:
    st.sidebar.warning("🔒 Vorbehandlung gesperrt (Pro)")
    pre_contrast = 1.0; pre_brightness = 0
    st.sidebar.info("Stil: Standard (Künstlerisch)")
    algo = "Künstlerisch (Skizze)"
    p_thresh = st.sidebar.slider("Helligkeit anpassen", 100, 255, 235, 5)
    p_blur = 55; p_noise = 50; p_erode = 0
    opt_light = False; opt_invert = False; opt_bg_weg = True 

# --- HAUPTBEREICH ---
uploaded_file = st.file_uploader("Bild hochladen", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # Tracking: Upload
    if 'last_upload' not in st.session_state or st.session_state['last_upload'] != uploaded_file.name:
        log_event("Bild Upload", "Neues Bild geladen")
        st.session_state['last_upload'] = uploaded_file.name

    image_pil = Image.open(uploaded_file)
    
    if is_pro:
        st.subheader("1. Zuschneiden (Pro Feature)")
        working_image_pil = st_cropper(image_pil, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        st.markdown("---")
    else:
        st.subheader("1. Original Bild")
        st.image(image_pil, caption="Original (Zuschneiden nur in Pro)", width=400)
        st.caption("🔒 Zuschneiden ist ein Pro-Feature")
        working_image_pil = image_pil

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Verarbeitung")
        with st.spinner('Arbeite...'):
            raw_img = bild_laden_und_freistellen(working_image_pil, opt_bg_weg)
            
        if raw_img is not None:
            adjusted_img = bild_vorbehandeln(raw_img, pre_brightness, pre_contrast)
            st.image(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("3. Ergebnis")
        if raw_img is not None:
            result_bin = erstelle_skizze_live(adjusted_img, algo, p_blur, p_thresh, p_noise, p_erode, opt_invert, opt_light)
            
            if not is_pro:
                st.caption("Free Version Vorschau")
            
            st.image(result_bin, caption="Ergebnis", use_container_width=True)
            
            svg_data = erstelle_svg_datei(result_bin)
            if svg_data:
                if is_pro:
                    with st.expander("🔍 Vektor-Vorschau (Pro)"):
                        st.markdown(render_svg_html(svg_data), unsafe_allow_html=True)
                else:
                     with st.expander("🔍 Vektor-Vorschau (Pro)"):
                        st.warning("🔒 Vektor-Vorschau nur in der Pro Version sichtbar.")

                # Button Logik separat, damit wir den Klick tracken können
                btn = st.download_button("⬇️ SVG Download", svg_data, "plot.svg", "image/svg+xml", type="primary")
                
                if btn:
                    log_event("Download", f"SVG heruntergeladen (Pro: {is_pro})")

import streamlit as st
import cv2
import numpy as np
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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Plotter Converter", page_icon="✂️", layout="wide")

# --- 2. LOGGER FUNCTION (Saves data to Excel/CSV) ---
LOG_FILE = "usage_data.csv"

def log_event(event_type, detail):
    """Writes a row to the CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists to determine if we need a header
    file_exists = os.path.isfile(LOG_FILE)
    
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Event", "Detail"]) # Header
            
            writer.writerow([timestamp, event_type, detail])
    except Exception as e:
        print(f"Logging Error: {e}")

# --- 3. HELPER FUNCTIONS (Image Processing) ---

def preprocess_image(image, brightness, contrast):
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

def enhance_image_clahe(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def load_and_remove_bg(pil_image, remove_background):
    try:
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        input_data = img_byte_arr.getvalue()
        
        if remove_background:
            # --- LAZY IMPORT ---
            from rembg import remove 
            
            output_data = remove(input_data)
            pil_image = Image.open(io.BytesIO(output_data)).convert("RGBA")
            white_bg = Image.new("RGBA", pil_image.size, "WHITE")
            white_bg.paste(pil_image, (0, 0), pil_image)
            final_image = white_bg.convert("RGB")
        else:
            final_image = pil_image.convert("RGB")
            
        return np.array(final_image)[:, :, ::-1].copy()
    except Exception as e:
        st.error(f"Processing Error: {e}")
        return None

def create_sketch_live(image, algorithm, blur, thresh, noise, erode, invert, light_fix):
    if light_fix: image = enhance_image_clahe(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if algorithm == "Artistic (Sketch)":
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
    clean = np.zeros_like(inv_bin)
    for cnt in contours:
        if cv2.contourArea(cnt) > noise: cv2.drawContours(clean, [cnt], -1, 255, -1)
    
    final_result = cv2.bitwise_not(clean)
    if invert: return cv2.bitwise_not(final_result)
    return final_result

def create_svg_file(binary_image):
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp_bmp:
        cv2.imwrite(tmp_bmp.name, binary_image)
        bmp_path = tmp_bmp.name
    svg_path = bmp_path.replace(".bmp", ".svg")
    try:
        subprocess.run(["potrace", bmp_path, "-s", "--turdsize", "10", "--alphamax", "1.0", "-o", svg_path], stdout=subprocess.DEVNULL, check=True)
        with open(svg_path, "r", encoding="utf-8") as f: svg_content = f.read()
        if os.path.exists(bmp_path): os.remove(bmp_path)
        if os.path.exists(svg_path): os.remove(svg_path)
        return svg_content
    except Exception: return None

def render_svg_html(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" style="max-width: 100%; border: 1px solid #ddd; padding: 10px;"/>'


# --- 4. MAIN PROGRAM ---

# Tracking: Log page visits (happens on every reload)
if 'has_logged_start' not in st.session_state:
    log_event("Visit", "Page opened / reloaded")
    st.session_state['has_logged_start'] = True

# --- UI LOGIC (FREEMIUM) ---
st.sidebar.title("💎 Select Version")
app_mode = st.sidebar.radio("Status:", ["Free Version", "Pro Version (Upgrade)"])

# Tracking: When user switches mode
if 'last_mode' not in st.session_state or st.session_state['last_mode'] != app_mode:
    log_event("Mode Switch", f"Switched to: {app_mode}")
    st.session_state['last_mode'] = app_mode

is_pro = (app_mode == "Pro Version (Upgrade)")

st.title("✂️ Plotter Converter")
if not is_pro:
    st.info("You are using the **Free Version**. Upgrade to **Pro** to unlock Cropping, Light Correction, and Advanced Filters!")
else:
    st.success("✨ **Pro Version** active. All features unlocked!")

# --- ADMIN AREA (Hidden in sidebar at the bottom) ---
st.sidebar.markdown("---")
with st.sidebar.expander("🔐 Admin / Analytics"):
    password = st.text_input("Password", type="password")
    if password == "study2024": # <--- YOUR PASSWORD
        if os.path.exists(LOG_FILE):
            st.write("📊 Usage Data:")
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df)
            
            with open(LOG_FILE, "rb") as f:
                st.download_button("📥 Download CSV File", f, file_name="usage_data.csv")
        else:
            st.warning("No data available yet.")

# --- CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Settings")

if is_pro:
    with st.sidebar.expander("🛠️ Pre-processing (Pro)", expanded=True):
        pre_contrast = st.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
        pre_brightness = st.slider("Brightness", -100, 100, 0, 5)
    
    algo = st.sidebar.radio("Style:", ["Artistic (Sketch)", "Scanner (Adaptive)"])
    p_blur = st.sidebar.slider("Detail Level", 1, 151, 55, 2)
    p_thresh = st.sidebar.slider("Threshold", 0, 255, 235, 1)
    p_noise = st.sidebar.slider("Remove Noise", 0, 200, 50, 1)
    p_erode = st.sidebar.slider("Thicken Lines", 0, 3, 0, 1)
    
    st.sidebar.subheader("Options")
    opt_light = st.sidebar.checkbox("💡 Auto Light Balance", value=True)
    opt_bg_weg = st.sidebar.checkbox("Remove Background", value=True)
    opt_invert = st.sidebar.checkbox("Invert Colors", value=False)
else:
    st.sidebar.warning("🔒 Pre-processing locked (Pro)")
    pre_contrast = 1.0; pre_brightness = 0
    st.sidebar.info("Style: Standard (Artistic)")
    algo = "Artistic (Sketch)"
    p_thresh = st.sidebar.slider("Adjust Brightness", 100, 255, 235, 5)
    p_blur = 55; p_noise = 50; p_erode = 0
    opt_light = False; opt_invert = False; opt_bg_weg = True 

# --- MAIN AREA ---
uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # Tracking: Upload
    if 'last_upload' not in st.session_state or st.session_state['last_upload'] != uploaded_file.name:
        log_event("Image Upload", "New image uploaded")
        st.session_state['last_upload'] = uploaded_file.name

    image_pil = Image.open(uploaded_file)
    
    if is_pro:
        st.subheader("1. Crop Image (Pro Feature)")
        working_image_pil = st_cropper(image_pil, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        st.markdown("---")
    else:
        st.subheader("1. Original Image")
        st.image(image_pil, caption="Original (Cropping only in Pro)", width=400)
        st.caption("🔒 Cropping is a Pro-Feature")
        working_image_pil = image_pil

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Processing")
        with st.spinner('Processing...'):
            raw_img = load_and_remove_bg(working_image_pil, opt_bg_weg)
            
        if raw_img is not None:
            adjusted_img = preprocess_image(raw_img, pre_brightness, pre_contrast)
            st.image(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("3. Result")
        if raw_img is not None:
            result_bin = create_sketch_live(adjusted_img, algo, p_blur, p_thresh, p_noise, p_erode, opt_invert, opt_light)
            
            if not is_pro:
                st.caption("Free Version Preview")
            
            st.image(result_bin, caption="Result", use_container_width=True)
            
            svg_data = create_svg_file(result_bin)
            
            if svg_data:
                # 1. Preview (Feature Gating)
                if is_pro:
                    with st.expander("🔍 Vector Preview (Pro)"):
                        st.markdown(render_svg_html(svg_data), unsafe_allow_html=True)
                else:
                     with st.expander("🔍 Vector Preview (Pro)"):
                        st.warning("🔒 Vector preview visible in Pro Version only.")

                # 2. Logging Callback Logic
                def log_download_click():
                    mode_text = "PRO" if is_pro else "FREE"
                    detail_text = f"SVG Download started (Mode: {mode_text})"
                    log_event("Download", detail_text)

                # 3. The Button with Callback (on_click)
                st.download_button(
                    label="⬇️ SVG Download", 
                    data=svg_data, 
                    file_name="plot.svg", 
                    mime="image/svg+xml", 
                    type="primary",
                    on_click=log_download_click 
                )

# --- FOOTER / PRIVACY ---
st.markdown("---")
st.caption("🔒 **Privacy Notice:** Your images are processed in memory only and are not stored permanently. They disappear once you close the browser. Only anonymous usage statistics (clicks) are recorded for this study project.")

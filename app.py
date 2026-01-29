# app.py — Artistic Image Processor with Enhanced UI
"""
Streamlit DIP app — Enhanced UI with beautiful design
- All original DIP functions kept
- Modern, attractive UI with gradient design
- Fixed text color contrast issues
- Enhanced visual feedback
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io, os, time, base64
from skimage import restoration
from skimage.util import random_noise

# optional: matplotlib for FFT/hist
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_OK = True
except Exception as e:
    MATPLOTLIB_OK = False

st.set_page_config(
    page_title="🎨 Artistic Image Processor", 
    layout="wide",
    page_icon="🎨",
    initial_sidebar_state="expanded"
)

# ----------------- Enhanced Modern Styling -----------------
# Replace your current styling block with this one
st.markdown(
    """
    <style>
    /* Page background — subtle light gradient */
    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 100%) !important;
        min-height: 100vh;
    }

    /* Main content container — white card for best contrast */
    .main .block-container {
        background: #ffffff !important;
        border-radius: 16px;
        padding: 2rem !important;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15,23,42,0.04);
        color: #0b1720 !important; /* very dark text for readability */
    }

    /* App header & subtitle (dark text) */
    .app-header {
        color: #0b1720 !important;
        font-weight: 800;
        font-size: 2.6rem;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        text-align: center;
        color: #263244 !important;
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }

    /* Headings and paragraph text in main area */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0b1720 !important;
    }

    /* Sidebar: soft indigo background with readable off-white text */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2b3656 0%, #2f3e63 100%) !important;
        color: #eef2ff !important;
        padding: 1.2rem !important;
    }
    [data-testid="stSidebar"] * {
        color: #eef2ff !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Sidebar sections look slightly translucent */
    .sidebar-section {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 10px;
        padding: 10px;
        margin: 0.6rem 0;
        border: 1px solid rgba(255,255,255,0.04);
    }

    /* Buttons — bluish with white text and subtle shadow */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 18px rgba(59,130,246,0.18) !important;
        transition: transform .18s ease, box-shadow .18s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(59,130,246,0.20) !important;
    }

    /* File uploader contrast */
    .stFileUploader > div {
        background: rgba(255,255,255,0.98) !important;
        border: 2px dashed rgba(15,23,42,0.06) !important;
        border-radius: 12px !important;
        padding: 18px !important;
    }

    /* Image cards */
    .image-card { background: #ffffff !important; border-radius: 14px; padding: 1.2rem; box-shadow: 0 10px 30px rgba(15,23,42,0.06); }
    .image-card * { color: #0b1720 !important; }

    .image-frame { border-radius: 12px; overflow: hidden; box-shadow: 0 8px 26px rgba(15,23,42,0.06); margin: 1rem 0; }

    /* Metric cards — subtle tint */
    .metric-card { background: linear-gradient(90deg, rgba(99,102,241,0.06), rgba(59,130,246,0.04)); border-radius: 10px; padding: 10px; text-align:center; }
    .metric-value { color: #0b1720 !important; font-weight:800; font-size:1.45rem; }
    .metric-label { color: #334155 !important; }

    /* Alerts readability */
    .stAlert { color: #0b1720 !important; background: #fff !important; border-radius: 10px; border-left-width: 6px !important; }

    /* Tabs / Expanders / Inputs contrast */
    .streamlit-expanderHeader { background: rgba(15,23,42,0.02) !important; color: #0b1720 !important; border-radius: 8px !important; }
    .stSelectbox [data-baseweb="select"] div, .stTextInput input { color: #0b1720 !important; }
    .stSlider span { color: #0b1720 !important; }

    /* Footer */
    .app-footer { text-align:center; color: #374151 !important; padding: 1.2rem; margin-top: 2rem; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(15,23,42,0.12); border-radius: 10px; }

    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------- Constants -----------------
PREVIEW_MAX_WIDTH = 720
PREVIEW_MAX_HEIGHT = 600

# ----------------- Helper Functions (unchanged) -----------------
def read_image_bytes(data_bytes):
    arr = np.frombuffer(data_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imwrite_rgb(path, img_rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def ensure_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def resize_for_preview_keep_aspect(img_rgb, max_w=PREVIEW_MAX_WIDTH, max_h=PREVIEW_MAX_HEIGHT):
    h, w = img_rgb.shape[:2]
    scale_w = max_w / w if w > max_w else 1.0
    scale_h = max_h / h if h > max_h else 1.0
    scale = min(scale_w, scale_h)
    if scale >= 1.0:
        return img_rgb
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

def pil_from_rgb(img_rgb):
    return Image.fromarray(img_rgb)

# ----------------- DIP functions (unchanged - keep all original functions) -----------------
def add_gaussian_noise(img, mean=0, var=0.01):
    imgf = img.astype(np.float32)/255.0
    out = random_noise(imgf, mode='gaussian', mean=mean, var=var, clip=True)
    return (out*255).astype(np.uint8)

def add_poisson_noise(img):
    imgf = img.astype(np.float32)/255.0
    out = random_noise(imgf, mode='poisson', clip=True)
    return (out*255).astype(np.uint8)

def add_uniform_noise(img, low=-0.05, high=0.05):
    imgf = img.astype(np.float32)/255.0
    noise = np.random.uniform(low, high, imgf.shape)
    out = np.clip(imgf + noise, 0, 1.0)
    return (out*255).astype(np.uint8)

def add_exponential_noise(img, scale=0.02):
    imgf = img.astype(np.float32)/255.0
    noise = np.random.exponential(scale, imgf.shape)
    out = np.clip(imgf + noise, 0, 1.0)
    return (out*255).astype(np.uint8)

def add_gamma_noise(img, shape_k=2.0, scale_theta=0.02):
    imgf = img.astype(np.float32)/255.0
    noise = np.random.gamma(shape_k, scale_theta, imgf.shape)
    out = np.clip(imgf + noise, 0, 1.0)
    return (out*255).astype(np.uint8)

def add_salt_pepper(img, amount=0.02, s_vs_p=0.5):
    out = img.copy()
    h, w = out.shape[:2]
    num_salt = int(np.ceil(amount * h * w * s_vs_p))
    ys = np.random.randint(0, h, num_salt)
    xs = np.random.randint(0, w, num_salt)
    out[ys, xs] = 255
    num_pepper = int(np.ceil(amount * h * w * (1 - s_vs_p)))
    ys = np.random.randint(0, h, num_pepper)
    xs = np.random.randint(0, w, num_pepper)
    out[ys, xs] = 0
    return out

def gaussian_blur(img, k=5):
    return cv2.GaussianBlur(img, (k,k), 0)

def median_blur(img, k=5):
    return cv2.medianBlur(img, k)

def bilateral_filter(img, d=9):
    return cv2.bilateralFilter(img, d, sigmaColor=75, sigmaSpace=75)

def wiener_filter_gray(img_gray, mysize=5):
    imgf = img_gray.astype(np.float64)/255.0
    den = restoration.wiener(imgf, psf=np.ones((mysize,mysize))/(mysize**2), balance=0.1)
    den = np.clip(den*255, 0, 255).astype(np.uint8)
    return den

def nl_means_denoise(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    return den[:,:,::-1]

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def unsharp_mask(img, strength=1.0, ksize=5):
    blurred = cv2.GaussianBlur(img, (ksize,ksize), 0)
    sharpened = cv2.addWeighted(img, 1+strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def dehaze_simple(img, patch=15):
    imgf = img.astype(np.float32)/255.0
    dark = cv2.erode(np.min((imgf*255).astype(np.uint8), axis=2),
                     cv2.getStructuringElement(cv2.MORPH_RECT,(patch,patch)))
    h,wid = dark.shape
    num = max(1,int(0.001*h*wid))
    idx = np.argpartition(dark.ravel(), -num)[-num:]
    yx = np.unravel_index(idx, dark.shape)
    brightest = imgf[yx]
    A = np.max(brightest, axis=0)
    transmission = 1 - 0.95*(dark.astype(np.float32)/255.0)
    transmission = np.clip(transmission, 0.1, 1.0)
    transmission = np.repeat(transmission[:,:,None], 3, axis=2)
    J = (imgf - A) / transmission + A
    return np.clip(J*255, 0, 255).astype(np.uint8)

def change_brightness_contrast(img, brightness=1.0, contrast=1.0):
    pil = Image.fromarray(img)
    if brightness != 1.0:
        pil = ImageEnhance.Brightness(pil).enhance(brightness)
    if contrast != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(contrast)
    return np.array(pil)

def rotate_img(img, angle_deg):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REPLICATE)

def translate_img(img, tx, ty):
    h,w = img.shape[:2]
    M = np.float32([[1,0,tx],[0,1,ty]])
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REPLICATE)

def scale_img(img, sx, sy):
    h,w = img.shape[:2]
    return cv2.resize(img, (max(1,int(w*sx)), max(1,int(h*sy))), interpolation=cv2.INTER_LINEAR)

def plot_fft_magnitude(img_rgb):
    if not MATPLOTLIB_OK:
        return None
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift)+1e-8)
    fig = Figure(figsize=(6,4))
    ax = fig.subplots()
    ax.imshow(magnitude_spectrum, cmap='inferno')
    ax.set_title("FFT Magnitude Spectrum")
    ax.axis('off')
    fig.patch.set_facecolor('white')
    return fig

def plot_color_histogram(img_rgb):
    if not MATPLOTLIB_OK:
        return None
    fig = Figure(figsize=(8,3))
    ax = fig.subplots()
    chans = ('r','g','b')
    colors = ['#FF5252', '#4CAF50', '#2196F3']
    for i,(col,color) in enumerate(zip(chans, colors)):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0,256])
        ax.plot(hist, color=color, label=col.upper(), linewidth=2, alpha=0.8)
    ax.set_xlim([0,256])
    ax.set_title("RGB Color Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor('white')
    return fig

# ----------------- NEW: Cartoonizer & Pencil Sketch functions -----------------
def _kmeans_color_quantization(img, k=8, attempts=8):
    Z = img.reshape((-1,3)).astype(np.float32)
    k = max(2, int(k))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, label, center = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        img_quant = res.reshape(img.shape)
        return img_quant
    except Exception:
        return img

def cartoonize(img_rgb, num_bilateral=5, bilateral_d=9, k_colors=8, edge_block=9, edge_C=2, edge_dilate=1):
    img = img_rgb.copy()
    for _ in range(max(1,int(num_bilateral))):
        img = cv2.bilateralFilter(img, d=int(bilateral_d), sigmaColor=75, sigmaSpace=75)
    img_quant = _kmeans_color_quantization(img, k=max(2,int(k_colors)))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    block = int(edge_block)
    if block % 2 == 0: block = max(3, block+1)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=block, C=int(edge_C))
    if int(edge_dilate) > 0:
        kernel = np.ones((int(edge_dilate), int(edge_dilate)), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(img_quant, edges_col)
    return cartoon

def dodge_blend(front, back):
    denom = 255 - back
    denom[denom == 0] = 1
    result = (front.astype(np.float32) * 255.0 / denom).clip(0,255).astype(np.uint8)
    return result

def pencil_sketch(img_rgb, sigma=21, use_adaptive=True, adaptive_block=9, adaptive_C=7):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (0,0), float(sigma))
    sketch = dodge_blend(gray, blur)
    if use_adaptive:
        block = int(adaptive_block)
        if block % 2 == 0: block = max(3, block+1)
        sketch = cv2.adaptiveThreshold(sketch, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize=block, C=int(adaptive_C))
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

def apply_sepia(img):
    kernel = np.array([[0.393,0.769,0.189],
                       [0.349,0.686,0.168],
                       [0.272,0.534,0.131]])
    sep = cv2.transform(img.astype(np.float32), kernel)
    sep = np.clip(sep, 0, 255).astype(np.uint8)
    return sep

# ----------------- Instagram-like filters -----------------
def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def filter_vintage(img):
    sep = apply_sepia(img)
    out = cv2.addWeighted(img, 0.6, sep, 0.4, 0)
    out = cv2.convertScaleAbs(out, alpha=1.05, beta=-10)
    return out

def filter_lomo(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 255)
    bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    high = cv2.convertScaleAbs(bright, alpha=1.2, beta=5)
    h,w = high.shape[:2]
    X = cv2.getGaussianKernel(w, w//1)
    Y = cv2.getGaussianKernel(h, h//1)
    kernel = Y * X.T
    mask = kernel / kernel.max()
    mask = cv2.merge([mask, mask, mask])
    out = (high.astype(np.float32) * mask + high.astype(np.float32)*(1-mask)*0.85).astype(np.uint8)
    return out

def filter_high_contrast(img):
    out = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    return out

def filter_vignette(img):
    h,w = img.shape[:2]
    X = cv2.getGaussianKernel(w, int(w*0.6))
    Y = cv2.getGaussianKernel(h, int(h*0.6))
    kernel = Y * X.T
    mask = kernel / kernel.max()
    mask = cv2.merge([mask, mask, mask])
    out = (img.astype(np.float32) * mask + 20*(1-mask)).astype(np.uint8)
    return out

def filter_warm(img):
    out = img.astype(np.float32).copy()
    out[:,:,0] = np.clip(out[:,:,0] * 1.05, 0, 255)
    out[:,:,1] = np.clip(out[:,:,1] * 1.02, 0, 255)
    out[:,:,2] = np.clip(out[:,:,2] * 0.95, 0, 255)
    return out.astype(np.uint8)

def filter_cool(img):
    out = img.astype(np.float32).copy()
    out[:,:,0] = np.clip(out[:,:,0] * 0.95, 0, 255)
    out[:,:,1] = np.clip(out[:,:,1] * 1.02, 0, 255)
    out[:,:,2] = np.clip(out[:,:,2] * 1.05, 0, 255)
    return out.astype(np.uint8)

def filter_film(img):
    imgf = img.astype(np.float32)
    grain = (np.random.randn(*img.shape) * 8).astype(np.int16)
    out = np.clip(imgf + grain, 0, 255).astype(np.uint8)
    out = cv2.convertScaleAbs(out, alpha=0.95, beta=5)
    return out

INSTAGRAM_FILTERS = {
    "None": lambda x: x,
    "B&W": to_grayscale,
    "Vintage": filter_vintage,
    "Lomo": filter_lomo,
    "High Contrast": filter_high_contrast,
    "Vignette": filter_vignette,
    "Warm": filter_warm,
    "Cool": filter_cool,
    "Film": filter_film
}

# ----------------- Defaults & session keys -----------------
DEFAULTS = {
    "noise_option":"None","g_mean":0.0,"g_var":0.01,
    "u_low":-0.05,"u_high":0.05,"e_scale":0.02,"g_k":2.0,"g_theta":0.02,
    "sp_amt":0.02,"sp_vs":0.5,
    "filter_option":"None","k":5,
    "do_unsharp":False,"un_strength":1.0,
    "do_clahe":False,"do_dehaze":False,
    "do_rotate":False,"angle":0,
    "do_translate":False,"tx":0,"ty":0,
    "do_scale":False,"sx":1.0,"sy":1.0,
    "flip_h":False,"flip_v":False,
    "brightness":1.0,"contrast":1.0,
    "show_fft":False,"show_hist":False,
    "instagram_filter":"None",
    "style_mode":"None",
    "cart_num_bilateral":5, "cart_bilateral_d":9, "cart_k_colors":8, "cart_edge_block":9, "cart_edge_C":2, "cart_edge_dilate":1,
    "sk_sigma":21, "sk_adaptive":True, "sk_block":9, "sk_C":7
}

if st.session_state.get("_reset_requested", False):
    for k,v in DEFAULTS.items(): st.session_state[k] = v
    st.session_state.pop("processed_img", None)
    st.session_state.pop("_reset_requested", None)

for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if 'uploaded_bytes' not in st.session_state:
    st.session_state.uploaded_bytes = None
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None
if 'history' not in st.session_state:
    st.session_state.history = []

# ----------------- Enhanced Sidebar -----------------
with st.sidebar:
    # Sidebar Header
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: white; font-size: 1.8rem; margin-bottom: 0.5rem;'>🎨 Artistic Processor</h1>
            <p style='color: #e2e8f0; font-size: 0.9rem;'>Transform your images with advanced filters</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Image Upload Section
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader("Drag & drop or click to browse", 
                                     type=["png","jpg","jpeg"],
                                     help="Upload your image for processing",
                                     label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.session_state.uploaded_bytes = uploaded_file.read()
        st.session_state.processed_img = None
        st.session_state.history = []
        st.success("✅ Image uploaded successfully!")
    
    # Quick Actions Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Load Sample", use_container_width=True, help="Load sample image from sample_images folder"):
            sd = "sample_images"
            if os.path.exists(sd):
                files = [f for f in os.listdir(sd) if f.lower().endswith((".png",".jpg",".jpeg"))]
                if files:
                    with open(os.path.join(sd, files[0]), "rb") as f:
                        st.session_state.uploaded_bytes = f.read()
                    st.success(f"✅ Loaded: {files[0]}")
                    st.session_state.processed_img = None
                    st.session_state.history = []
                else:
                    st.warning("No images in sample_images/")
            else:
                st.info("Create 'sample_images/' folder and add images")
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True, help="Reset all settings to default"):
            for k,v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.processed_img = None
            st.session_state.history = []
            st.experimental_rerun()
    
    st.markdown("---")
    
    # Noise Section
    with st.expander("🎭 Noise Effects", expanded=False):
        st.selectbox("Noise Type", ["None","Gaussian","Poisson","Uniform","Exponential","Gamma","Salt & Pepper"], key="noise_option")
        if st.session_state["noise_option"] == "Gaussian":
            st.slider("Mean", 0.0, 0.5, key="g_mean", step=0.01, help="Mean value for Gaussian noise")
            st.slider("Variance", 0.0001, 0.1, key="g_var", step=0.0001, help="Variance for Gaussian noise")
        elif st.session_state["noise_option"] == "Uniform":
            st.slider("Low", -0.2, 0.0, key="u_low", step=0.01)
            st.slider("High", 0.0, 0.2, key="u_high", step=0.01)
        elif st.session_state["noise_option"] == "Exponential":
            st.slider("Scale", 0.001, 0.2, key="e_scale", step=0.001)
        elif st.session_state["noise_option"] == "Gamma":
            st.slider("Shape k", 0.5, 10.0, key="g_k", step=0.1)
            st.slider("Scale theta", 0.001, 0.1, key="g_theta", step=0.001)
        elif st.session_state["noise_option"] == "Salt & Pepper":
            st.slider("Amount", 0.0, 0.2, key="sp_amt", step=0.005, help="Density of salt & pepper noise")
            st.slider("Salt vs Pepper", 0.0, 1.0, key="sp_vs", step=0.01, help="Ratio of salt to pepper")
    
    st.markdown("---")
    
    # Filters & Enhancement Section
    with st.expander("✨ Filters & Enhancement", expanded=False):
        st.selectbox("Filter Type", ["None","Gaussian","Median","Bilateral","Wiener(gray)","NonLocalMeans"], key="filter_option")
        st.selectbox("Kernel Size", [3,5,7,9], key="k")
        
        st.checkbox("Unsharp Mask", key="do_unsharp", help="Enhance edges for sharper images")
        if st.session_state["do_unsharp"]:
            st.slider("Strength", 0.1, 3.0, key="un_strength", step=0.1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("CLAHE", key="do_clahe", help="Contrast Limited Adaptive Histogram Equalization")
        with col2:
            st.checkbox("Dehaze", key="do_dehaze", help="Remove haze/fog from images")
    
    st.markdown("---")
    
    # Augmentation Section
    with st.expander("🔄 Image Augmentation", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Rotate", key="do_rotate")
        with col2:
            if st.session_state["do_rotate"]:
                st.slider("Angle", -180, 180, key="angle", step=5)
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Translate", key="do_translate")
        with col2:
            if st.session_state["do_translate"]:
                st.slider("X", -200, 200, key="tx", step=5)
        
        if st.session_state["do_translate"]:
            st.slider("Y", -200, 200, key="ty", step=5)
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Scale", key="do_scale")
        with col2:
            if st.session_state["do_scale"]:
                st.slider("Scale X", 0.2, 2.0, key="sx", step=0.1)
        
        if st.session_state["do_scale"]:
            st.slider("Scale Y", 0.2, 2.0, key="sy", step=0.1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Flip Horizontal", key="flip_h")
        with col2:
            st.checkbox("Flip Vertical", key="flip_v")
        
        st.slider("Brightness", 0.2, 2.0, key="brightness", step=0.1, help="Adjust image brightness")
        st.slider("Contrast", 0.2, 2.0, key="contrast", step=0.1, help="Adjust image contrast")
    
    st.markdown("---")
    
    # Instagram Filters Section
    with st.expander("🎨 Instagram Filters", expanded=False):
        st.selectbox("Select Filter", list(INSTAGRAM_FILTERS.keys()), key="instagram_filter")
        st.markdown("<small style='color: #e2e8f0;'>Apply artistic filters for creative effects</small>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cartoon & Sketch Section
    with st.expander("🎨 Cartoon & Sketch", expanded=False):
        st.selectbox("Art Style", ["None","Cartoon","PencilSketch"], key="style_mode")
        
        if st.session_state["style_mode"] == "Cartoon":
            st.markdown("#### Cartoon Parameters")
            st.slider("Bilateral Iterations", 1, 12, key="cart_num_bilateral", help="More iterations = smoother colors")
            st.slider("Bilateral Diameter", 3, 15, step=2, key="cart_bilateral_d", help="Pixel neighborhood size")
            st.slider("Color Clusters", 2, 24, key="cart_k_colors", help="Number of colors in final image")
            st.slider("Edge Detail", 3, 21, step=2, key="cart_edge_block", help="Block size for edge detection")
            st.slider("Edge Strength", 0, 10, key="cart_edge_C", help="Higher values = stronger edges")
            st.slider("Edge Thickness", 0, 5, key="cart_edge_dilate", help="Make edges thicker")
        
        elif st.session_state["style_mode"] == "PencilSketch":
            st.markdown("#### Sketch Parameters")
            st.slider("Blur Intensity", 1, 60, key="sk_sigma", help="Controls the softness of sketch lines")
            st.checkbox("Ink Effect", key="sk_adaptive", help="Create dramatic ink sketch look")
            if st.session_state["sk_adaptive"]:
                st.slider("Detail Level", 3, 31, step=2, key="sk_block", help="Higher = more detailed sketch")
                st.slider("Contrast", -10, 20, key="sk_C", help="Adjust sketch contrast")
    
    st.markdown("---")
    
    # Analysis Section
    with st.expander("📊 Analysis Tools", expanded=False):
        st.checkbox("Show FFT Magnitude", key="show_fft", help="Visualize frequency domain")
        st.checkbox("Show Color Histogram", key="show_hist", help="Analyze color distribution")
    
    st.markdown("---")
    
    # Save & History Section
    st.markdown("### 💾 Save & History")
    outdir = st.text_input("Output Folder", value="outputs", help="Folder where processed images will be saved")
    
    col_save, col_undo = st.columns(2)
    with col_save:
        if st.button("💾 Save Image", use_container_width=True, help="Save processed image to output folder"):
            if st.session_state.get("processed_img", None) is not None:
                fname = os.path.join(outdir, f"processed_{int(time.time())}.png")
                try:
                    imwrite_rgb(fname, st.session_state.processed_img)
                    st.success(f"✅ Saved: {fname}")
                except Exception as e:
                    st.error("❌ Save failed: " + str(e))
            else:
                st.error("⚠️ No processed image to save")
    
    with col_undo:
        if st.button("↩️ Undo Last", use_container_width=True, help="Undo last transformation"):
            if st.session_state.history:
                st.session_state.processed_img = st.session_state.history.pop()
                st.success("✅ Undo applied")
            else:
                st.info("ℹ️ No history to undo")

# ----------------- Main Application Area -----------------
st.markdown('<h1 class="app-header">🎨 Artistic Image Processor</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Transform your photos with advanced digital image processing techniques</p>', unsafe_allow_html=True)

# Check if image is loaded
if not st.session_state.uploaded_bytes:
    # Welcome screen with sample previews
    st.markdown("### 📤 Get Started")
    st.info("Upload an image using the sidebar or load a sample image to begin processing.")
    
    # Show sample previews if folder exists
    if os.path.exists("sample_images"):
        sample_files = [f for f in os.listdir("sample_images") if f.lower().endswith((".png",".jpg",".jpeg"))]
        if sample_files:
            st.markdown("### Sample Images Available:")
            cols = st.columns(min(4, len(sample_files)))
            for idx, file in enumerate(sample_files[:4]):
                with cols[idx]:
                    try:
                        with open(os.path.join("sample_images", file), "rb") as f:
                            img_bytes = f.read()
                            img = read_image_bytes(img_bytes)
                            img_resized = resize_for_preview_keep_aspect(img, max_w=200, max_h=150)
                            st.image(img_resized, caption=file, use_column_width=True)
                    except:
                        pass
    
    # Features showcase
    st.markdown("---")
    st.markdown("### ✨ Features")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">🎭</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Noise Effects</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">✨</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Filters</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">🎨</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Art Styles</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Analysis</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# Load and process image
try:
    img_orig = read_image_bytes(st.session_state.uploaded_bytes)
except Exception as e:
    st.error("❌ Failed to read image: " + str(e))
    st.stop()

img_orig = ensure_rgb(img_orig)

# Build processed image via DIP pipeline
proc = img_orig.copy()

# Apply noise if selected
try:
    nopt = st.session_state.get("noise_option","None")
    if nopt != "None":
        if nopt == "Gaussian":
            proc = add_gaussian_noise(proc, mean=st.session_state.get("g_mean",0.0), var=st.session_state.get("g_var",0.01))
        elif nopt == "Poisson":
            proc = add_poisson_noise(proc)
        elif nopt == "Uniform":
            proc = add_uniform_noise(proc, low=st.session_state.get("u_low",-0.05), high=st.session_state.get("u_high",0.05))
        elif nopt == "Exponential":
            proc = add_exponential_noise(proc, scale=st.session_state.get("e_scale",0.02))
        elif nopt == "Gamma":
            proc = add_gamma_noise(proc, shape_k=st.session_state.get("g_k",2.0), scale_theta=st.session_state.get("g_theta",0.02))
        elif nopt == "Salt & Pepper":
            proc = add_salt_pepper(proc, amount=st.session_state.get("sp_amt",0.02), s_vs_p=st.session_state.get("sp_vs",0.5))
except Exception as e:
    st.warning("⚠️ Noise application failed: " + str(e))

# Apply filters
try:
    fopt = st.session_state.get("filter_option","None")
    kval = int(st.session_state.get("k",5))
    if fopt == "Gaussian":
        proc = gaussian_blur(proc, kval)
    elif fopt == "Median":
        proc = median_blur(proc, kval)
    elif fopt == "Bilateral":
        proc = bilateral_filter(proc, d=kval*2+1)
    elif fopt == "Wiener(gray)":
        gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
        den = wiener_filter_gray(gray, mysize=kval)
        proc = cv2.cvtColor(den, cv2.COLOR_GRAY2RGB)
    elif fopt == "NonLocalMeans":
        proc = nl_means_denoise(proc)
except Exception as e:
    st.warning("⚠️ Filter application failed: " + str(e))

# Apply enhancements
try:
    if st.session_state.get("do_unsharp",False):
        proc = unsharp_mask(proc, strength=st.session_state.get("un_strength",1.0), ksize=3)
    if st.session_state.get("do_clahe",False):
        proc = apply_clahe(proc)
    if st.session_state.get("do_dehaze",False):
        proc = dehaze_simple(proc, patch=15)
except Exception as e:
    st.warning("⚠️ Enhancement failed: " + str(e))

# Apply augmentations
try:
    if st.session_state.get("do_rotate",False) and st.session_state.get("angle",0) != 0:
        proc = rotate_img(proc, st.session_state.get("angle",0))
    if st.session_state.get("do_translate",False) and (st.session_state.get("tx",0) != 0 or st.session_state.get("ty",0) != 0):
        proc = translate_img(proc, st.session_state.get("tx",0), st.session_state.get("ty",0))
    if st.session_state.get("do_scale",False) and (st.session_state.get("sx",1.0) != 1.0 or st.session_state.get("sy",1.0) != 1.0):
        proc = scale_img(proc, st.session_state.get("sx",1.0), st.session_state.get("sy",1.0))
    if st.session_state.get("flip_h",False):
        proc = cv2.flip(proc, 1)
    if st.session_state.get("flip_v",False):
        proc = cv2.flip(proc, 0)
    if st.session_state.get("brightness",1.0) != 1.0 or st.session_state.get("contrast",1.0) != 1.0:
        proc = change_brightness_contrast(proc, brightness=st.session_state.get("brightness",1.0), contrast=st.session_state.get("contrast",1.0))
except Exception as e:
    st.warning("⚠️ Augmentation failed: " + str(e))

# Apply Instagram-style filter
inst = st.session_state.get("instagram_filter","None")
if inst and inst in INSTAGRAM_FILTERS:
    try:
        proc = INSTAGRAM_FILTERS[inst](proc)
    except Exception as e:
        st.warning("⚠️ Instagram filter failed: " + str(e))

# Apply Cartoon/Sketch style
style = st.session_state.get("style_mode", "None")
if style == "Cartoon":
    try:
        proc = cartoonize(
            proc,
            num_bilateral=st.session_state.get("cart_num_bilateral",5),
            bilateral_d=st.session_state.get("cart_bilateral_d",9),
            k_colors=st.session_state.get("cart_k_colors",8),
            edge_block=st.session_state.get("cart_edge_block",9),
            edge_C=st.session_state.get("cart_edge_C",2),
            edge_dilate=st.session_state.get("cart_edge_dilate",1)
        )
    except Exception as e:
        st.warning("⚠️ Cartoonize failed: " + str(e))
elif style == "PencilSketch":
    try:
        proc = pencil_sketch(
            proc,
            sigma=st.session_state.get("sk_sigma",21),
            use_adaptive=st.session_state.get("sk_adaptive",True),
            adaptive_block=st.session_state.get("sk_block",9),
            adaptive_C=st.session_state.get("sk_C",7)
        )
    except Exception as e:
        st.warning("⚠️ Pencil sketch failed: " + str(e))

# Store processed image
st.session_state.processed_img = proc.copy()

# Prepare previews
left_preview = resize_for_preview_keep_aspect(img_orig)
right_preview = resize_for_preview_keep_aspect(st.session_state.processed_img)

# Show side-by-side images in cards
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Original Image")
    
    # Active filters badges
    active_filters = []
    if st.session_state["noise_option"] != "None":
        active_filters.append(f'🎭 {st.session_state["noise_option"]}')
    if st.session_state["filter_option"] != "None":
        active_filters.append(f'✨ {st.session_state["filter_option"]}')
    if st.session_state["instagram_filter"] != "None":
        active_filters.append(f'🎨 {st.session_state["instagram_filter"]}')
    if st.session_state["style_mode"] != "None":
        active_filters.append(f'🖼️ {st.session_state["style_mode"]}')
    
    if active_filters:
        badges_html = "".join([f'<span class="badge badge-secondary">{f}</span>' for f in active_filters])
        st.markdown(f'<div style="margin-bottom: 1rem;">{badges_html}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    st.image(left_preview, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Image info
    st.markdown(f"""
    <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(102, 126, 234, 0.05); border-radius: 8px;">
        <div style="display: flex; justify-content: space-between;">
            <span style="font-weight: 600; color: #2d3748;">Dimensions:</span>
            <span style="color: #2d3748;">{img_orig.shape[1]} × {img_orig.shape[0]}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.25rem;">
            <span style="font-weight: 600; color: #2d3748;">Preview:</span>
            <span style="color: #2d3748;">{left_preview.shape[1]} × {left_preview.shape[0]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.markdown("### ✨ Processed Image")
    
    # Status indicator
    processing_status = "🟢 Processing complete"
    if any([st.session_state["noise_option"] != "None",
            st.session_state["filter_option"] != "None",
            st.session_state["instagram_filter"] != "None",
            st.session_state["style_mode"] != "None"]):
        processing_status = "⚡ Processing applied"
    
    st.markdown(f'<div style="margin-bottom: 1rem; color: #48BB78;"><strong>{processing_status}</strong></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    st.image(right_preview, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processed image info
    st.markdown(f"""
    <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(72, 187, 120, 0.05); border-radius: 8px;">
        <div style="display: flex; justify-content: space-between;">
            <span style="font-weight: 600; color: #2d3748;">Dimensions:</span>
            <span style="color: #2d3748;">{proc.shape[1]} × {proc.shape[0]}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.25rem;">
            <span style="font-weight: 600; color: #2d3748;">Preview:</span>
            <span style="color: #2d3748;">{right_preview.shape[1]} × {right_preview.shape[0]}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.25rem;">
            <span style="font-weight: 600; color: #2d3748;">History:</span>
            <span style="color: #2d3748;">{len(st.session_state.history)} steps</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Image Statistics
st.markdown("---")
st.markdown("### 📊 Image Statistics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{img_orig.shape[0]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Height (px)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{img_orig.shape[1]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Width (px)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    channels = "RGB" if img_orig.shape[2] == 3 else "Grayscale"
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{channels}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Color Mode</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(st.session_state.history)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Undo Steps</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis Visualizations
if st.session_state.get("show_fft", False) or st.session_state.get("show_hist", False):
    st.markdown("---")
    st.markdown("### 🔬 Analysis Tools")
    
    if st.session_state.get("show_fft", False):
        if MATPLOTLIB_OK:
            st.markdown("#### Frequency Analysis (FFT)")
            fig = plot_fft_magnitude(st.session_state.processed_img)
            if fig: 
                st.pyplot(fig)
                st.markdown("*FFT magnitude spectrum showing frequency distribution*", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Matplotlib not available — FFT visualization disabled")
    
    if st.session_state.get("show_hist", False):
        if MATPLOTLIB_OK:
            st.markdown("#### Color Distribution")
            fig = plot_color_histogram(st.session_state.processed_img)
            if fig: 
                st.pyplot(fig)
                st.markdown("*RGB color histogram showing pixel intensity distribution*", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Matplotlib not available — histogram visualization disabled")

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="app-footer">
        <p>🎨 <strong>Artistic Image Processor</strong> | Powered by OpenCV & Streamlit</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem; color: rgba(255, 255, 255, 0.8);">
            Adjust parameters in the sidebar to fine-tune your image transformations
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
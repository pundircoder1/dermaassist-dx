import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from timm import create_model
from PIL import Image
import numpy as np
import base64
import io
from groq import Groq
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="DermaAssist-DX", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body { font-family: 'DM Sans', sans-serif !important; background-color: #0A0D14 !important; }
.stApp { background: #0A0D14 !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1400px !important; }

.hero { padding: 40px 0 28px 0; border-bottom: 1px solid #1E2330; margin-bottom: 32px; }
.hero-badge {
    display: inline-block; background: #0F3460; color: #4DA6FF;
    font-size: 15px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    padding: 8px 18px; border-radius: 3px; margin-bottom: 18px; border: 1px solid #1A4A80;
}
.hero-title {
    font-size: 72px; font-weight: 300; letter-spacing: -3px; line-height: 1;
    color: #FFFFFF; margin: 0 0 14px 0;
}
.hero-title b { font-weight: 700; color: #4DA6FF; }
.hero-sub { font-size: 22px; color: #6B7280; font-weight: 400; }

.upload-lbl {
    font-size: 15px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    color: #4B5563; margin-bottom: 8px; display: block;
}

.ph { display:flex; align-items:center; gap:12px; margin-bottom:20px; padding-bottom:16px; border-bottom:1px solid #1A2035; }
.pi { width:38px; height:38px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:20px; }
.pi-b { background:#0F2A4A; } .pi-t { background:#0A2A2A; } .pi-a { background:#2A1A0A; }
.pt { font-size: 17px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #9CA3AF; }

.pb {
    background: linear-gradient(135deg,#0F2A4A,#0A1F38); border: 1px solid #1A4A80;
    border-radius: 8px; padding: 22px 26px; margin: 16px 0;
}
.pb.lc { background: linear-gradient(135deg,#2A1A0A,#1F1308); border-color:#5A3A0A; }
.plbl { font-size: 15px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #4B5563; margin-bottom: 10px; }
.pcls { font-size: 32px; font-weight: 700; color: #FFFFFF; line-height: 1.2; }
.pconf { font-family: 'DM Mono', monospace; font-size: 22px; color: #4DA6FF; margin-top: 8px; }
.pconf.lc { color: #F59E0B; }

.btitle { font-size: 15px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #4B5563; margin: 22px 0 16px 0; }
.bwrap { margin-bottom: 16px; }
.blbl { display:flex; justify-content:space-between; font-size: 19px; color: #6B7280; margin-bottom: 7px; }
.bpct { font-family: 'DM Mono', monospace; font-size: 19px; color: #9CA3AF; }
.btrack { height: 6px; background: #1A2035; border-radius: 3px; overflow: hidden; }
.bfill     { height:100%; border-radius:3px; background: linear-gradient(90deg,#1A4A80,#4DA6FF); }
.bfill.r2  { background: linear-gradient(90deg,#0A3A3A,#2DD4BF); }
.bfill.r3  { background: linear-gradient(90deg,#2A1A0A,#F59E0B); }

.wb { background:#1A0F00; border:1px solid #5A3A0A; border-left:4px solid #F59E0B; border-radius:4px; padding:18px 20px; margin:14px 0; }
.wt { font-size: 19px; font-weight: 700; color: #F59E0B; margin-bottom: 8px; }
.wb2 { font-size: 18px; color: #9CA3AF; line-height: 1.65; }

.rs { margin-bottom: 22px; }
.rst {
    font-size: 15px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    color: #4DA6FF; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}
.rst::before { content:''; display:inline-block; width:18px; height:1px; background:#4DA6FF; flex-shrink:0; }
.rb { font-size: 19px; color: #C9D1E0; line-height: 1.8; }
.rb ul { padding-left: 22px; margin: 10px 0; }
.rb li { margin-bottom: 8px; }
.rdiv { height:1px; background:#1A2035; margin:18px 0; }

.hl { display:flex; align-items:center; gap:12px; margin-top:16px; padding:14px 16px; background:#080C17; border-radius:6px; border:1px solid #1A2035; }
.hbar { width:72px; height:11px; border-radius:3px; background:linear-gradient(90deg,#00008B,#0000FF,#00FFFF,#FFFF00,#FF8000,#FF0000); flex-shrink:0; }
.hlbl { font-size: 17px; color: #6B7280; }
.hcap { font-size: 18px; color: #4B5563; line-height: 1.7; margin-top: 16px; }

.sdiv { height:1px; background:linear-gradient(90deg,transparent,#1E2A3D,transparent); margin:28px 0; }
.empty { text-align:center; padding:70px 0; }
.eico { font-size: 52px; opacity: 0.2; margin-bottom: 16px; }
.etxt { font-size: 20px; color: #4B5563; font-weight: 600; }
.esub { font-size: 18px; color: #374151; margin-top: 8px; }

.foot { display:flex; align-items:center; justify-content:space-between; margin-top:52px; padding-top:22px; border-top:1px solid #1A2035; }
.fl { font-size: 16px; color: #374151; }
.fr { font-family: 'DM Mono', monospace; font-size: 14px; color: #1F2937; letter-spacing: 1px; }

.stImage > img { border-radius: 8px; border: 1px solid #1A2035; }
[data-testid="stFileUploaderDropzone"] { background:#080C17 !important; border:1.5px dashed #1E2A3D !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH   = 'best_model_v4.pth'
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
IMG_SIZE     = 384

CLASS_NAMES = [
    'Infectious Disorders','Inflammatory Disorders','Keratanisation Disorders',
    'Neoplasms and tumors','No Definite Diagnosis','Other skin disorders',
    'Pigmentary Disorders','Skin Appendages Disorders'
]
CLASS_ICONS = {
    'Infectious Disorders':'🦠','Inflammatory Disorders':'🔥','Keratanisation Disorders':'📋',
    'Neoplasms and tumors':'🔬','No Definite Diagnosis':'❓','Other skin disorders':'🩹',
    'Pigmentary Disorders':'🎨','Skin Appendages Disorders':'💇',
}
IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD  = [0.229,0.224,0.225]

@st.cache_resource
def load_model():
    m = create_model('swin_base_patch4_window12_384', pretrained=False, num_classes=8, img_size=384)
    ck = torch.load(MODEL_PATH, map_location='cpu')
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def swin_reshape(t, h=12, w=12):
    if len(t.shape)==4: return t.permute(0,3,1,2)
    return t.reshape(t.size(0),h,w,t.size(2)).permute(0,3,1,2)

def predict(model, img):
    inp = tfm(img).unsqueeze(0)
    with torch.no_grad():
        out=model(inp); probs=F.softmax(out,dim=1)[0]
        conf=probs.max().item(); pred=probs.argmax().item()
    top3=[(CLASS_NAMES[i],probs[i].item()) for i in probs.topk(3).indices.tolist()]
    return CLASS_NAMES[pred], conf, top3, inp

def get_gradcam(model, inp, pred_idx):
    cam = GradCAM(model=model, target_layers=[model.layers[-1].blocks[-1].norm1], reshape_transform=swin_reshape)
    gc  = cam(input_tensor=inp, targets=[ClassifierOutputTarget(pred_idx)])[0]
    rgb = np.array(inp[0].permute(1,2,0))
    rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
    return show_cam_on_image(rgb.astype(np.float32), gc, use_rgb=True)

def get_reasoning(image, pred_class, confidence, top3):
    client = Groq(api_key=GROQ_API_KEY)
    buf    = io.BytesIO()
    image.save(buf, format='JPEG', quality=85)
    b64    = base64.standard_b64encode(buf.getvalue()).decode('utf-8')
    others = [c for c,_ in top3[1:]]
    prompt = f"""A dermatology AI model predicted: {pred_class} ({confidence:.1%} confidence). Other possibilities: {', '.join(others)}.

Generate a clinical reasoning report with exactly these 4 sections:
**VISUAL FEATURES:** What visible characteristics support this prediction?
**CLINICAL REASONING:** Why is this consistent with the observed features?
**DIFFERENTIAL DIAGNOSIS:** What else should be considered?
**RECOMMENDED ACTION:** Specific next steps for the patient.

Use bullet points. Frame as 'the model suggests'. 2-3 bullets per section."""
    resp = client.chat.completions.create(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        messages=[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}","detail":"low"}},
            {"type":"text","text":prompt}
        ]}], max_tokens=500)
    return resp.choices[0].message.content

# ── UI ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Clinical AI &nbsp;·&nbsp; Research Tool</div>
  <div class="hero-title">Derma<b>Assist</b>-DX</div>
  <div class="hero-sub">AI-powered dermatological analysis for Indian skin conditions</div>
</div>
""", unsafe_allow_html=True)

model = load_model()

st.markdown('<span class="upload-lbl">Upload Skin Image</span>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop image here", type=['jpg','jpeg','png','avif','webp','bmp','tiff'], label_visibility='collapsed')

if not uploaded:
    st.markdown('<div class="empty"><div class="eico">🩺</div><div class="etxt">No image uploaded</div><div class="esub">Supports JPG, PNG, AVIF, WebP and more</div></div>', unsafe_allow_html=True)
else:
    img = Image.open(uploaded).convert('RGB')
    with st.spinner('Running analysis...'):
        pred_class, confidence, top3, inp = predict(model, img)
        pred_idx  = CLASS_NAMES.index(pred_class)
        pred_icon = CLASS_ICONS.get(pred_class,'🔬')
        high_conf = confidence >= 0.5

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1.1], gap="medium")

    with col1:
        st.markdown('<div class="ph"><div class="pi pi-b">📷</div><span class="pt">Source Image</span></div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        lc = "lc" if not high_conf else ""
        st.markdown(f"""
        <div class="pb {lc}">
          <div class="plbl">Model Prediction</div>
          <div class="pcls">{pred_icon}&nbsp; {pred_class}</div>
          <div class="pconf {lc}">{confidence:.1%} confidence</div>
        </div>""", unsafe_allow_html=True)
        if not high_conf:
            st.markdown('<div class="wb"><div class="wt">⚠ Low Confidence</div><div class="wb2">Below 50% threshold. Clinical reasoning suppressed. Please consult a dermatologist.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="btitle">Top Predictions</div>', unsafe_allow_html=True)
        for i,(cls,prob) in enumerate(top3):
            pct=int(prob*100); rc=["","r2","r3"][i]
            st.markdown(f'<div class="bwrap"><div class="blbl"><span>{cls}</span><span class="bpct">{pct}%</span></div><div class="btrack"><div class="bfill {rc}" style="width:{pct}%"></div></div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ph"><div class="pi pi-t">🔍</div><span class="pt">Grad-CAM Attention</span></div>', unsafe_allow_html=True)
        with st.spinner('Computing heatmap...'):
            heatmap = get_gradcam(model, inp, pred_idx)
        st.image(heatmap, use_container_width=True)
        st.markdown('<div class="hl"><div class="hbar"></div><div class="hlbl">Low &nbsp;→&nbsp; High attention &nbsp;(red = strongest focus)</div></div><div class="hcap">Grad-CAM highlights skin regions that most influenced the model\'s prediction. Clinically meaningful focus on lesion morphology rather than background artifacts.</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="ph"><div class="pi pi-a">🧠</div><span class="pt">Clinical Reasoning</span></div>', unsafe_allow_html=True)
        if high_conf:
            with st.spinner('Generating clinical report...'):
                reasoning = get_reasoning(img, pred_class, confidence, top3)
            sections = {
                'VISUAL FEATURES':        ('Visual Features','🔎'),
                'CLINICAL REASONING':     ('Clinical Reasoning','🧬'),
                'DIFFERENTIAL DIAGNOSIS': ('Differential Diagnosis','⚖️'),
                'RECOMMENDED ACTION':     ('Recommended Action','📋'),
            }
            for key,(title,icon) in sections.items():
                marker = f'**{key}:**'
                if marker in reasoning:
                    parts = reasoning.split(marker)
                    rest  = parts[1] if len(parts)>1 else ''
                    end   = len(rest)
                    for ok in sections:
                        om = f'**{ok}:**'
                        if ok!=key and om in rest:
                            idx=rest.index(om)
                            if idx<end: end=idx
                    content = rest[:end].strip()
                    st.markdown(f'<div class="rs"><div class="rst">{icon}&nbsp; {title}</div><div class="rb">{content}</div></div><div class="rdiv"></div>', unsafe_allow_html=True)
            if not any(f'**{k}:**' in reasoning for k in sections):
                st.markdown(f'<div class="rb">{reasoning}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="wb" style="margin-bottom:24px"><div class="wt">Clinical reasoning unavailable</div><div class="wb2">Confidence below 50% threshold. Generating clinical text for uncertain predictions risks misleading outputs.</div></div><div class="rs"><div class="rst">📋&nbsp; Recommended Action</div><div class="rb"><ul><li>Consult a qualified dermatologist for evaluation</li><li>Retake with better lighting and focus on the lesion</li><li>Ensure lesion is centred and clearly visible</li><li>Use original camera quality — avoid compression</li></ul></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="foot"><div class="fl">⚠&nbsp; Research tool only — not validated for clinical use. Always consult a qualified dermatologist.</div><div class="fr">UPES Dehradun &nbsp;·&nbsp; B-11 &nbsp;·&nbsp; DermaCon-IN &nbsp;·&nbsp; Swin-Base 384</div></div>', unsafe_allow_html=True)

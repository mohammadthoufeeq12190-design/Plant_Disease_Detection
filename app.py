import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from model import PlantDiseaseCNN

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f4fff5;
}

h1{
    color:#2E8B57;
}

h2,h3{
    color:#2E8B57;
}

div[data-testid="metric-container"]{
    background:#E8F5E9;
    border-radius:12px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL FROM HUGGING FACE
# -------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="thoufeeqmohd/plant-disease-cnn-model",
        filename="plant_disease_model.pth"
    )

    model = PlantDiseaseCNN()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


model = load_model()

# -------------------------------------------------
# CLASSES
# -------------------------------------------------

classes = [
    "Diseased",
    "Healthy"
]

# -------------------------------------------------
# IMAGE TRANSFORM
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.image("assets/logo.png", width=150)

st.sidebar.title("Plant Disease Detection")

st.sidebar.markdown("""
### About

This AI application detects whether a plant leaf is healthy or diseased using a Convolutional Neural Network.

### Technologies

- Python
- PyTorch
- CNN
- Streamlit
""")

st.sidebar.success("Model Loaded Successfully")

# -------------------------------------------------
# HEADER
# -------------------------------------------------

col1, col2 = st.columns([1,5])

with col1:

    st.image("assets/logo.png", width=120)

with col2:

    st.title("🌿 Plant Disease Detection System")

    st.write(
        "AI-powered Plant Leaf Disease Detection using Deep Learning (CNN)"
    )

st.divider()

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

c1,c2,c3,c4 = st.columns(4)

c1.metric("Classes","2")
c2.metric("Model","CNN")
c3.metric("Framework","PyTorch")
c4.metric("Interface","Streamlit")

st.divider()

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left,right = st.columns(2)

    with left:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.subheader("Image Details")

        st.write(f"Width : {image.width}px")
        st.write(f"Height : {image.height}px")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(image_tensor)

        probability = F.softmax(output,dim=1)

        confidence,predicted = torch.max(probability,1)

    prediction = classes[predicted.item()]

    confidence = confidence.item()*100

    healthy_probability = probability[0][1].item()*100
    diseased_probability = probability[0][0].item()*100

    with right:

        st.subheader("Prediction")

        if prediction=="Healthy":

            st.success("🌿 Healthy Plant")

        else:

            st.error("🍂 Diseased Plant")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

        st.subheader("Prediction Probabilities")

        st.write(f"🌿 Healthy : {healthy_probability:.2f}%")
        st.progress(int(healthy_probability))

        st.write(f"🍂 Diseased : {diseased_probability:.2f}%")
        st.progress(int(diseased_probability))

        if prediction=="Healthy":

            st.info("""
### Recommendation

✅ Continue watering regularly.

✅ Give enough sunlight.

✅ Monitor leaves weekly.

✅ Apply fertilizer when needed.

✅ Plant looks healthy.
""")

        else:

            st.warning("""
### Recommendation

⚠ Disease detected.

⚠ Remove infected leaves.

⚠ Avoid overwatering.

⚠ Apply suitable fungicide.

⚠ Keep infected plants away from healthy plants.
""")

st.divider()

st.markdown("""
<center>

## 🌿 Plant Disease Detection using CNN

Built with ❤️ using

**Python • PyTorch • Streamlit**

Developed by **Mohammad Thoufeeq**

</center>
""", unsafe_allow_html=True)
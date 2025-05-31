import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load your trained YOLO model
model_path = 'f142e3ad-8214-4867-8c00-40b4040fbf64.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = YOLO(model_path)

# Set page configuration
st.set_page_config(page_title="Persons Detection Model", layout="centered")

# Sidebar Info
st.sidebar.title("🔖 About")
st.sidebar.markdown("""
**MUHAMMAD FARHAN RANA**  
📧 mfarhanrana982@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/muhammadfarhanrana)
""")

# App title
st.title("👤 Persons Detection Model")
st.markdown("Upload an image, and the trained (YOLOv8) model will detect **persons** in it.")

# Upload file
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name

    # Run detection
    with st.spinner("Detecting persons..."):
        results = model(image_path)
        boxes_img = results[0].plot()  # Plot boxes

    st.image(boxes_img, caption="Detection Results", use_column_width=True)

    # Show detected classes
    st.subheader("Detected Objects")
    for box in results[0].boxes.data.tolist():
        class_id = int(box[5])
        class_name = results[0].names[class_id]
        conf = float(box[4])
        st.write(f"🟩 {class_name}: {conf:.2f}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using [Ultralytics YOLOv8](https://docs.ultralytics.com/) and Streamlit.")


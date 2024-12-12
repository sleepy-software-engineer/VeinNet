import sys
import streamlit as st
from PIL import Image
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils.config import PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED
from utils.functions import split, mapping
from dataloader import DataLoader 
import torch
from model import Model
from typing import Tuple

def load_model(num_classes: int) -> Tuple[Model, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes)
    model.load_state_dict(torch.load("./model/model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def init_model() -> Tuple[Model, torch.device, DataLoader]:
    mapping_ids = mapping(PATIENTS)
    split_data = split(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)
    test_loader = DataLoader(split_data, "test", mapping_ids)
    num_classes = len(mapping_ids)
    model, device = load_model(num_classes)
    return model, device, test_loader

def selecte_image(model: Model, loader: DataLoader, device: torch.device) -> None:
    image_names = [os.path.basename(path) for path in loader.image_paths]
    selected_image_name = st.selectbox("Select an Image", image_names) 
    if selected_image_name:
        selected_image_path = next(path for path in loader.image_paths if os.path.basename(path) == selected_image_name)
        selected_image = Image.open(selected_image_path)
        vein_image, label, patient_id = loader.generate_image(selected_image_path)
        col1, col2 = st.columns(2)
        with col1:
            st.image(selected_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(vein_image, caption="Vein Image", use_container_width=True)
        with torch.no_grad():
            vein_image = torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            label = torch.tensor([int(label)], dtype=torch.long).to(device)
            output = model(vein_image)
            
        expected_label = loader.id_mapping[patient_id]
        predicted_label = str(torch.argmax(output, dim=1).item())
        
        container_color = "green" if int(expected_label) == int(predicted_label) else "red"
        with st.container():
            st.markdown(
            f"""
            <div style="background-color: {container_color}; padding: 10px; border-radius: 5px; text-align: center;">
                <span style="color: white; font-weight: bold;">Expected Label: {expected_label}</span>
                <span style="color: white; margin-left: 20px; font-weight: bold;">Predicted Label: {predicted_label}</span>
            </div>
            """,
            unsafe_allow_html=True
            )

def classify_all(model: Model, loader: DataLoader, device: torch.device) -> None:
    results = []
    for image_path in loader.image_paths:
        #selected_image = Image.open(image_path)
        vein_image, label, patient_id = loader.generate_image(image_path)
        with torch.no_grad():
            vein_image = torch.tensor(vein_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            label = torch.tensor([int(label)], dtype=torch.long).to(device)
            output = model(vein_image)
        expected_label = loader.id_mapping[patient_id]
        predicted_label = str(torch.argmax(output, dim=1).item())
        # expected_label = 0
        # predicted_label = 0
        match = "PASS" if int(expected_label) == int(predicted_label) else "FAIL"
        results.append((image_path, expected_label, predicted_label, match))
    
    st.table(results)

def main() -> None:
    st.title("VeinVision")
    st.subheader("A CNN-based Hand Vein Recognition System")
    model, device, loader = init_model()
    st.sidebar.title("Menu")
    option = st.sidebar.selectbox("Select a page", ["Select Image", "Classify All"])
    if option == "Select Image":
        selecte_image(model, loader, device)
    if option == "Classify All":
        classify_all(model, loader, device)
    
if __name__ == "__main__":
    main() 
    
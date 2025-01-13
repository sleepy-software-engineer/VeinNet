import os
import sys
from typing import Tuple

import streamlit as st
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

from identification.closed.dataloader import DataLoader as IdentificationDataLoader
from identification.model import Model as IdentificationModel
from utils.config import DATASET_PATH, HAND, PATIENTS, SEED, SPECTRUM
from utils.functions import (
    mapping,
    split_identification_closed,
    split_verification_closed,
)
from verification.dataloader import DataLoader as VerificationDataLoader
from verification.model import Model as VerificationModel


def load_verification_model(num_classes: int) -> Tuple[VerificationModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VerificationModel(num_classes)
    model.load_state_dict(
        torch.load(
            "./src/verification/model/model.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.to(device)
    model.eval()
    return model, device


def load_identification_model(
    num_classes: int,
) -> Tuple[IdentificationModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdentificationModel(num_classes)
    model.load_state_dict(
        torch.load(
            "./src/identification/closed/model/model.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.to(device)
    model.eval()
    return model, device


def init_verification_model() -> (
    Tuple[VerificationModel, torch.device, VerificationDataLoader]
):
    mapping_ids = mapping(PATIENTS)
    split_data = split_verification_closed(PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED)
    test_loader = VerificationDataLoader(split_data, "test", mapping_ids)
    num_classes = len(mapping_ids)
    model, device = load_verification_model(num_classes)
    return model, device, test_loader


def init_identification_model() -> (
    Tuple[IdentificationModel, torch.device, IdentificationDataLoader]
):
    mapping_ids = mapping(PATIENTS)
    split_data = split_identification_closed(
        PATIENTS, DATASET_PATH, HAND, SPECTRUM, SEED
    )
    test_loader = IdentificationDataLoader(split_data, "test", mapping_ids)
    num_classes = len(mapping_ids)
    model, device = load_identification_model(num_classes)
    return model, device, test_loader


def select_image(
    model: IdentificationModel, loader: IdentificationDataLoader, device: torch.device
) -> None:
    st.subheader(
        "This page allows you to select an image from the dataset and view the results."
    )
    image_names = ["Select an Image"] + [
        os.path.basename(path) for path in loader.image_paths
    ]
    selected_image_name = st.selectbox("Select an Image", image_names, index=0)
    if selected_image_name != "Select an Image":
        selected_image_path = next(
            path
            for path in loader.image_paths
            if os.path.basename(path) == selected_image_name
        )
        selected_image = Image.open(selected_image_path)
        vein_image, label, person_id = loader._generate_image(selected_image_path)
        col1, col2 = st.columns(2)
        with col1:
            st.image(selected_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(vein_image, caption="Vein Image", use_container_width=True)
        with torch.no_grad():
            vein_image = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            label = torch.tensor([int(label)], dtype=torch.long).to(device)
            output = model(vein_image)
        expected_label = f"Person-{person_id}"
        expected_label_id = loader.id_mapping[person_id]
        predicted_label_id = torch.argmax(output, dim=1).item()
        predicted_label = f"Person-{next(key for key, value in loader.id_mapping.items() if value == predicted_label_id)}"
        container_color = (
            "green" if int(expected_label_id) == int(predicted_label_id) else "red"
        )
        with st.container():
            st.markdown(
                f"""
            <div style="background-color: {container_color}; padding: 10px; border-radius: 5px; text-align: center;">
            <span style="color: white; font-weight: bold; font-size: 20px;">Expected: {expected_label}</span>
            <span style="color: white; margin-left: 20px; font-weight: bold; font-size: 20px;">Predicted: {predicted_label}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )


def classify_all(
    model: IdentificationModel, loader: IdentificationDataLoader, device: torch.device
) -> None:
    st.subheader(
        "This page allows you to classify all images in the dataset and view the results."
    )
    results = []
    for image_path in loader.image_paths:
        selected_image = Image.open(image_path)
        vein_image, label, person_id = loader._generate_image(image_path)
        with torch.no_grad():
            vein_image_tensor = (
                torch.tensor(vein_image, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            label = torch.tensor([int(label)], dtype=torch.long).to(device)
            output = model(vein_image_tensor)
        expected_label = f"Person-{person_id}"
        expected_label_id = loader.id_mapping[person_id]
        predicted_label_id = torch.argmax(output, dim=1).item()
        predicted_label = f"Person-{next(key for key, value in loader.id_mapping.items() if value == predicted_label_id)}"
        match = "MATCH" if int(expected_label_id) == predicted_label_id else "NO MATCH"
        match_color = "green" if match == "MATCH" else "red"
        results.append(
            (
                selected_image,
                vein_image,
                expected_label,
                predicted_label,
                match,
                match_color,
            )
        )
    st.write("Classification Results")
    if st.button("Classify All"):
        for result in results:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(result[0], use_container_width=True)
            with col2:
                st.image(result[1], use_container_width=True)
            with col3:
                st.write(result[2])
            with col4:
                st.write(result[3])
            with col5:
                st.markdown(
                    f"<span style='color: {result[5]}; font-weight: bold;'>{result[4]}</span>",
                    unsafe_allow_html=True,
                )


def verification(
    model: VerificationModel, loader: VerificationDataLoader, device: torch.device
) -> None:
    st.subheader(
        "This page allows you to visualize the verification process and results."
    )

    data_records = []
    with torch.no_grad():
        for vein_tensor, claim_label, is_genuine in loader.generate_data():
            prediction = model(vein_tensor, claim_label)
            predicted_label = torch.sigmoid(prediction).item() > 0.63
            is_correct = predicted_label == is_genuine.item()
            vein_image = vein_tensor.squeeze().cpu().numpy()
            data_records.append(
                {
                    "Input Image": vein_image,
                    "Claim Label": claim_label.item(),
                    "Is Genuine": is_genuine.item(),
                    "Model Prediction": predicted_label,
                    "Correct": is_correct,
                }
            )

    data_records.sort(key=lambda x: x["Claim Label"])
    st.write("Verification Results Table")
    for record in data_records:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(
                record["Input Image"],
                caption="Vein Image",
                use_container_width=True,
                clamp=True,
            )
        with col2:
            st.text(f"Claim Label: {record['Claim Label']}")
        with col3:
            st.text(f"Is Genuine: {record['Is Genuine']}")
        with col4:
            st.text(f"Prediction: {'True' if record['Model Prediction'] else 'False'}")
        with col5:
            correctness_color = "green" if record["Correct"] else "red"
            st.markdown(
                f"<span style='color: {correctness_color}; font-weight: bold;'>{'Correct' if record['Correct'] else 'Incorrect'}</span>",
                unsafe_allow_html=True,
            )


def main() -> None:
    st.title("VeinNet - CNN Biometric System")
    model, device, loader = init_identification_model()
    st.sidebar.title("Menu")
    option = st.sidebar.selectbox(
        "Select an option",
        ["Select Image to Classify", "Classify All Images", "Verification"],
    )
    if option == "Select Image to Classify":
        select_image(model, loader, device)
    if option == "Classify All Images":
        classify_all(model, loader, device)
    if option == "Verification":
        model, device, loader = init_verification_model()
        verification(model, loader, device)


if __name__ == "__main__":
    main()

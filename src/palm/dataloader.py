import glob
import cv2
import numpy as np
import os
from palm.dataprocessor import VeinImageProcessor
import matplotlib.pyplot as plt

DATASET_PATH = "/home/lucian/University/MSc-Courses/BiometricSystems/data/"
PATIENTS = [f"{i:03}" for i in range(1, 101)] 
HAND = "l"  
SPECTRUM = "940" 
BATCH_SIZE = 1
OUTPUT_DIR = "/home/lucian/University/MSc-Courses/BiometricSystems/src/palm/out/" 

class VeinImageDataLoader:
    def __init__(self, dataset_dir, batch_size=1):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.processor = VeinImageProcessor()  

    def _load_images(self, person_id, hand, spectrum):
        pattern = f"{person_id}_{hand}_{spectrum}_*.jpg"
        matching_files = glob.glob(os.path.join(self.dataset_dir, pattern))
        return matching_files

    def _prepare_cnn_input(self, images, target_size=(128, 128)):
        """
        Preprocess images for CNN input.
        Args:
            images (list): List of image paths.
            target_size (tuple): Target size for resizing images (height, width).
        Returns:
            np.ndarray: Preprocessed images in CNN-compatible format.
        """
        preprocessed_images = [self.processor.preprocess_image(img) for img in images]

        # Resize all images to the target size
        resized_images = [cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in preprocessed_images]

        # Stack images and add channel dimension
        stacked_images = np.stack(resized_images, axis=0)  # Shape: (num_images, height, width)
        cnn_input = np.expand_dims(stacked_images, axis=1)  # Add channel dimension: (num_images, 1, height, width)

        # Normalize pixel values to range [0, 1]
        cnn_input = cnn_input.astype(np.float32) / 255.0
        return cnn_input

    def generate_batches(self, hand, spectrum):
        """
        Generate batches of preprocessed images for multiple patients.
        Yields:
            Tuple[np.ndarray, List[str]]: (CNN input batch, list of patient IDs).
        """
        patient_batches = []
        patient_ids = []

        for person_id in PATIENTS:
            images = self._load_images(person_id, hand, spectrum)
            if len(images) == 6:  # Ensure exactly 6 images per patient
                cnn_input = self._prepare_cnn_input(images)
                patient_batches.append(cnn_input)
                patient_ids.append(person_id)

                if len(patient_batches) == self.batch_size:
                    yield np.stack(patient_batches, axis=0), patient_ids
                    patient_batches = []
                    patient_ids = []

        # Yield the last partial batch if it exists
        if patient_batches:
            yield np.stack(patient_batches, axis=0), patient_ids

def test_vein_image_dataloader():
    vein_loader = VeinImageDataLoader(dataset_dir=DATASET_PATH, batch_size=1)
    batch_generator = vein_loader.generate_batches(hand=HAND, spectrum=SPECTRUM)

    output_dir = os.path.join(OUTPUT_DIR, "test_images")
    os.makedirs(output_dir, exist_ok=True)

    for cnn_input, patient_ids in batch_generator:
        print(f"Patient ID: {patient_ids}")
        print(f"Input shape for CNN: {cnn_input.shape}")  # Expected: (1, 6, 1, 128, 128)

        # Process first batch for visualization
        for patient_idx, patient_id in enumerate(patient_ids):
            for img_idx in range(cnn_input.shape[1]):
                img = cnn_input[patient_idx, img_idx, 0]  # Get individual image (remove batch and channel dims)
                
                # Save the image
                save_path = os.path.join(output_dir, f"{patient_id}_img_{img_idx + 1}.png")
                plt.imsave(save_path, img, cmap='gray')
                print(f"Saved: {save_path}")
        break  # Test only the first batch
    
if __name__ == "__main__":
    test_vein_image_dataloader()
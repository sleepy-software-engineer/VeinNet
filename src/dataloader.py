import os
from dataprocessor import DataProcessor

class DataLoader:
    """
    DataLoader class for loading and processing biometric data.
    Attributes:
        image_paths (list): List of image file paths for the given data split.
        id_mapping (dict): Dictionary mapping patient IDs to their corresponding labels.
    """
    def __init__(self, split_data, split_name, id_mapping):
        self.image_paths = split_data[split_name]
        self.id_mapping = id_mapping
    
    def generate_data(self):
        for image_path in self.image_paths:
            patient_id = os.path.basename(image_path).split("_")[0]
            label = self.id_mapping[patient_id]
            processor = DataProcessor()
            vein_image = processor.preprocess_image(image_path)
            yield vein_image, label
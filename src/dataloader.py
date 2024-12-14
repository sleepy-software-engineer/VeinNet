import os

from dataprocessor import DataProcessor


class DataLoader:
    """
    DataLoader class for loading and processing biometric data.
    """

    def __init__(self, split_data, split_name, id_mapping):
        self.image_paths = split_data[split_name]
        self.id_mapping = id_mapping

    def generate_data(self):
        """
        Generates data by iterating over image paths and processing each image.
        """
        for image_path in self.image_paths:
            vein_image, label, _ = self.generate_image(image_path)
            yield vein_image, label

    def generate_image(self, image_path):
        """
        Generates a preprocessed vein image along with its label and patient ID.
        """
        processor = DataProcessor()
        patient_id = os.path.basename(image_path).split("_")[0]
        label = self.id_mapping[patient_id]
        vein_image = processor.preprocess_image(image_path)
        return vein_image, label, patient_id

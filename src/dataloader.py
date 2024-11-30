from hand.dataloader import HandGeometryDataLoader
from palm.dataloader import VeinImageDataLoader

DATASET_PATH = "/home/lucian/University/MSc-Courses/BiometricSystems/data/"
PATIENTS = [f"{i:03}" for i in range(1, 101)] 
HAND = "l"  
SPECTRUM = "940" 


class CombinedDataLoader:
    def __init__(self, vein_loader, geometry_loader, batch_size=1):
        """
        Initialize the combined data loader.
        Args:
            vein_loader: Instance of VeinImageDataLoader.
            geometry_loader: Instance of HandGeometryDataLoader.
            batch_size: Number of patients per batch.
        """
        self.vein_loader = vein_loader
        self.geometry_loader = geometry_loader
        self.batch_size = batch_size

    def generate_batches(self, hand, spectrum):
        """
        Generate combined batches of vein image data and hand geometry features.
        Args:
            hand: Hand type ('l' or 'r').
            spectrum: Spectrum type (e.g., '940').
        Yields:
            Tuple: (vein_image_batch, hand_geometry_batch, patient_ids)
        """
        # Generate batches from both dataloaders
        vein_batches = self.vein_loader.generate_batches(hand=hand, spectrum=spectrum)
        geometry_batches = self.geometry_loader.generate_batches(hand=hand, spectrum=spectrum)

        for (vein_image_batch, patient_ids_vein), (hand_geometry_batch, patient_ids_geometry) in zip(
            vein_batches, geometry_batches
        ):
            # Ensure synchronization of patient IDs
            assert patient_ids_vein == patient_ids_geometry, \
                f"Mismatch in patient IDs: {patient_ids_vein} != {patient_ids_geometry}"

            yield vein_image_batch, hand_geometry_batch, patient_ids_vein
def test_combined_dataloader():
    # Initialize individual dataloaders
    vein_loader = VeinImageDataLoader(dataset_dir=DATASET_PATH, batch_size=1)
    geometry_loader = HandGeometryDataLoader(dataset_dir=DATASET_PATH, batch_size=1)

    # Initialize the combined dataloader
    combined_loader = CombinedDataLoader(vein_loader, geometry_loader, batch_size=1)

    # Generate and test combined batches
    for vein_image_batch, hand_geometry_batch, patient_ids in combined_loader.generate_batches(hand=HAND, spectrum=SPECTRUM):
        print(f"Patient IDs: {patient_ids}")
        print(f"Vein Image Batch Shape: {vein_image_batch.shape}")  # Expected: (1, 6, 1, height, width)
        print(f"Hand Geometry Batch Shape: {hand_geometry_batch.shape}")  # Expected: (1, 6, num_features)       
        break  # Test only the first batch

if __name__ == "__main__":
    test_combined_dataloader()
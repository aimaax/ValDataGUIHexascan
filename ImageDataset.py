from torch.utils.data import Dataset
import os
import numpy as np
import torch


class ImageDataset(Dataset):
    def __init__(self, base_path_images, database, transform=None, only_non_validated=True):
        self.base_path_images = base_path_images
        self.database = database
        self.transform = transform
        
        # Calculate statistics and filter paths
        self._print_humanval_stats()
        self.image_paths = self._generate_image_paths(only_non_validated=only_non_validated)
        
        print("\n================================================= GUI Logger ==================================================\n")
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print("img path: ", img_path)
        
        image_numpy = np.load(img_path) # size (2736, 3840)
        if image_numpy.shape[0] == 2736:
            image_numpy = image_numpy[:-16, :] # size (2720, 3840)

        if self.transform:
            image_tensor = self.transform(image_numpy) # size (1, 2720, 3840)
        else:
            # If no transform, manually convert to tensor
            image_tensor = torch.from_numpy(image_numpy).float()
        return image_tensor, img_path

    def _generate_image_paths(self, only_non_validated = True):
        print("\n============================================ Configuring Dataloader ===========================================\n")
        print("Setting up all image paths to dataloader...")
        image_paths = set()
        
        # Reset the index to access columns easily
        records = self.database.reset_index()

        if only_non_validated == True:
            # Filter records where HumanVal = False
            records = records[records["HumanVal"] == False]
        
        for _, record in records.iterrows():
            campaign = record["Campaign"]
            dut = str(record["DUT"])
            filename = record["FileName"] + ".npy"

            # Construct the full path to the image directory
            image_dir = os.path.join(self.base_path_images, campaign, dut, filename)

            # Find .npy files in the directory
            if os.path.exists(image_dir):
                image_paths.add(image_dir)
                
        
        if not image_paths:
            raise ValueError("No valid image paths found in the dataset.")
        
        print("Image paths from database stored correctly! ")
        
        return list(image_paths)
    
    def _print_humanval_stats(self):
        """Print statistics about HumanVal and calculate the percentage of False."""
        total_entries = len(self.database)
        false_humanval = self.database[self.database["HumanVal"] == False].shape[0]
        true_humanval = total_entries - false_humanval
        
        # Calculate anomalies and normal entries based on 'x' value
        anomalies = self.database[self.database["x"] != "nan"].shape[0]
        normals = total_entries - anomalies

        # Calculate percentage of False HumanVal
        false_percentage = (false_humanval / total_entries) * 100 if total_entries > 0 else 0

        print("\n============================================ Validation Statistics ============================================\n")
        print(f"Total entries: {total_entries} | Anomalies: {anomalies} | Normal: {normals}")
        print(f"HumanVal = True: {true_humanval}")
        print(f"HumanVal = False: {false_humanval}")
        print(f"Percentage of entries left to validate: {false_percentage:.2f}%")
    
    def remove_image_path(self, img_path):
        """Remove an image path from the dataset."""
        if img_path in self.image_paths:
            self.image_paths.remove(img_path)
            # print(f"Image path {img_path} removed from dataset.")
        else:
            print(f"\nImage path {img_path} not found in dataset.")
        
import sys
import os
import torch
import pandas as pd
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QGridLayout, QHBoxLayout
)
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter, QPen
from PySide6.QtCore import Qt, QRect, QTimer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import GaussianBlur
from PIL import Image
import numpy as np
from ImageDataset import ImageDataset
from TrackingDataLoader import TrackingDataLoader
from scipy.ndimage import gaussian_filter
from torch import manual_seed



class GUI_HexaScan(QMainWindow):
    def __init__(self, dataloader, database, db_path_store, root_path_images, batch_size, test_boolean = False):
        super().__init__()
        self.dataloader = dataloader
        self.tracking_dataloader = TrackingDataLoader(self.dataloader)
        self.database = database
        self.db_path_store = db_path_store
        self.root_path_images = root_path_images
        self.batch_size = batch_size
        self.test_boolean = test_boolean
        window_scaling_factor = 0.8
        self.window_x = 1200 * window_scaling_factor
        self.window_y = 850 * window_scaling_factor
        self.batch_iterator = iter(self.tracking_dataloader)
        
        self.previous_batch = None  # Store only the last displayed batch
        self.current_batch = None  # Track the currently displayed batch
        self.last_iterated_batch = None  # Track the last batch fetched from the iterator
        self.viewing_previous = False  # Flag to indicate if viewing a previous batch
        self.removed_image_boolean = False # Track if an image has been removed
        self.annotations = []  # Store interactive annotations for current image
        
        self.grid_size = 160
        self.image_width = 3840
        self.image_height = 2720
        
        # Calculate grid dimensions for different image layouts with different batch sizes
        self.grid_rows = int(np.ceil(np.sqrt(batch_size)))
        self.grid_cols = int(np.ceil(batch_size / self.grid_rows))
        
        # Set up the main window
        self.setWindowTitle("HexaScan")
        self.resize(self.window_x, self.window_y)
        
        # Central widget and layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        
        # Add grid for images
        self.image_widgets = []
        for i in range(self.grid_rows * self.grid_cols):
            if i < self.batch_size:  # Only create widgets for valid batch size
                
                image_container = QGridLayout()  # Use QGridLayout for precise layout

                # Create the filename label
                filename_label = QLabel()
                filename_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Align left
                filename_label.setStyleSheet("color: black; font-weight: bold;")

                # Create the validation label
                validation_label = QLabel()
                validation_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # Align right

                if batch_size == 1:
                    filename_label.setFont(QFont("Arial", 10, QFont.Bold))
                    validation_label.setFont(QFont("Arial", 10, QFont.Bold))
                else:
                    filename_label.setFont(QFont("Arial", 8, QFont.Bold))
                    validation_label.setFont(QFont("Arial", 8, QFont.Bold))
                                    
                # Create the image label
                image_label = QLabel()
                image_label.setAlignment(Qt.AlignCenter)
                if batch_size == 1:
                    image_label.setMouseTracking(True)  # Enable mouse tracking
                    image_label.mousePressEvent = self.mousePressEvent  # Attach event handler

                # Add widgets to the container grid layout
                image_container.addWidget(filename_label, 0, 0)  # Filename label in row 0, col 0
                image_container.addWidget(validation_label, 0, 1)  # Validation label in row 0, col 1
                image_container.addWidget(image_label, 1, 0, 1, 2)  # Image spans two columns in row 1


                # Add the image container to the grid layout
                self.grid_layout.addLayout(image_container, i // self.grid_cols, i % self.grid_cols)

                # Store references to the widgets for later updates
                self.image_widgets.append((image_label, filename_label, validation_label))
                
                
        
        self.layout.addLayout(self.grid_layout)
        
        # Add Navigation Buttons
        self.button_layout = QHBoxLayout()  # Horizontal layout for buttons
        
        # Remove image from storage and database
        self.remove_button = QPushButton("Remove image")
        self.remove_button.clicked.connect(self.remove_image_from_database)
        self.remove_button.setAutoDefault(False)  # Disable auto-default
        self.remove_button.setDefault(False)
        self.remove_button.setFocusPolicy(Qt.NoFocus)
        if batch_size == 1:
            self.remove_button.setEnabled(True)
        else:
            self.remove_button.setEnabled(False)
            
        self.button_layout.addWidget(self.remove_button, stretch=1)
        
        # Previous Button
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.load_previous_batch)
        self.previous_button.setEnabled(False)  # Disabled initially
        self.button_layout.addWidget(self.previous_button, stretch=1)
        
        # Next Button
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.load_next_batch)
        self.next_button.setDefault(True)
        self.next_button.setFocus()
        self.button_layout.addWidget(self.next_button, stretch=2)
        
        self.layout.addLayout(self.button_layout)  # Add buttons to the main layout
        
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        # Load the first batch
        self.load_next_batch()
        
    

    def load_next_batch(self):
        if self.test_boolean == False:
            self.database.to_pickle(self.db_path_store)
        self.annotations = []
        if self.viewing_previous:  # If viewing a previous batch, resume from iterator
            self.viewing_previous = False
            self.display_batch(self.last_iterated_batch)
            self.previous_button.setEnabled(True)
            return

        try:
            # Save the current batch as the previous batch before loading the next one
            if self.current_batch is not None:
                self.previous_batch = self.current_batch
                if self.removed_image_boolean == True:
                    self.removed_image_boolean = False
                    self.previous_button.setEnabled(False) # No previous when removing an image
                else:    
                    self.previous_button.setEnabled(True)  # Enable Previous button
            
            batch = next(self.batch_iterator)
            self.current_batch = batch  # Update the current batch
            self.last_iterated_batch = batch  # Save the last iterated batch
            self.display_batch(batch)
        except StopIteration:
            self.previous_batch = None  # Clear previous batch
            self.previous_button.setEnabled(False)  # Disable Previous button
            self.display_noise_image() # Display noise image to know that it has been going around
            
            # Schedule loading the next batch after a short delay
            QTimer.singleShot(1000, self.reset_iterator_and_load_first_batch)
       
            
    def reset_iterator_and_load_first_batch(self):
        """Reset the iterator and load the first batch of the new round."""
        self.batch_iterator = iter(self.tracking_dataloader)
        batch = next(self.batch_iterator)
        self.current_batch = batch  # Reset the current batch
        self.display_batch(batch)


    def load_previous_batch(self):
        if self.test_boolean == False:
            self.database.to_pickle(self.db_path_store)
        self.annotations = []
        
        if self.previous_batch is not None:
            self.display_batch(self.previous_batch)  # Display the previous batch
            self.previous_button.setEnabled(False)
            self.viewing_previous = True  # Indicate that we are viewing a previous batch
            
    def remove_image_from_database(self):
        if self.viewing_previous==True:
            _, img_path = self.previous_batch
        else:
            _, img_path = self.current_batch
            

        components = img_path[0].split(os.sep)
        campaign = components[-3]
        dut = str(components[-2])
        filename = components[-1]
        
        # Create a condition to match the first three levels
        condition = (
            (self.database.index.get_level_values("Campaign") == campaign) &
            (self.database.index.get_level_values("DUT") == dut) &
            (self.database.index.get_level_values("FileName") == filename[:-4])
        )
        
        # self.database = self.database.drop((campaign, dut, filename)).sort_index()
        self.database = self.database[~condition].sort_index()
        
        # When removed, set previous batch to None and move to next batch
        self.previous_batch = None
        self.removed_image_boolean = True
        
        # Remove image path from dataloader to take it away from the iterator
        self.dataloader.dataset.remove_image_path(img_path[0])
        
        # Remove from disk
        if self.test_boolean == False:
            try:
                os.remove(img_path[0]) 
                print(f"\nDeleted file: {img_path[0]}")
            except FileNotFoundError:
                print(f"\nFile not found: {img_path[0]}")
            except Exception as e:
                print(f"\nError deleting file: {img_path[0]} - {e}")
        else: 
            print(f"\nIn test mode, did not remove file: {img_path[0]} from the disk.")
        
        current_index = self.tracking_dataloader.get_current_index()
        
        if self.viewing_previous == True:
            current_index -= 1
        self.viewing_previous = False

        # Refresh the DataLoader and batch iterator
        self.tracking_dataloader = TrackingDataLoader(DataLoader(
            self.dataloader.dataset,
            batch_size=self.batch_size,
            shuffle=False
        ))
        self.batch_iterator = iter(self.tracking_dataloader)
        self.tracking_dataloader.reset_to_position(current_index)
        
        self.load_next_batch()

    def display_batch(self, batch, reset_annotations=True):
        images, image_paths = batch

        if reset_annotations == True:
            self.annotations = [] # reset annotations
        
        for i, (image_tensor, img_path) in enumerate(zip(images, image_paths)):
            components = img_path.split(os.sep)
            campaign = components[-3]
            dut = str(components[-2])
            filename = components[-1]
            
            # Apply Gaussian filter to the image
            # print("image_tensor shape:", image_tensor.shape)
            gaussian_blur = GaussianBlur(kernel_size=(7, 7), sigma=(1.5, 1.5))
            image_tensor = gaussian_blur(image_tensor.unsqueeze(0)).squeeze(0).squeeze(0)
            image_numpy = (image_tensor.numpy() * 255).astype(np.uint8)
            image = QImage(image_numpy, image_numpy.shape[1], image_numpy.shape[0], QImage.Format_Grayscale8)
            
            
            # Retrieve annotation boxes for this filename
            if reset_annotations:
                condition = (
                    (self.database.index.get_level_values("Campaign") == campaign) &
                    (self.database.index.get_level_values("DUT") == dut) &
                    (self.database.index.get_level_values("FileName") == filename[:-4])
                )
                
                # Retrieve the current value of HumanVal
                self.current_human_val = self.database.loc[condition, "HumanVal"].iloc[0] if not self.database.loc[condition].empty else None
                
                # Update HumanVal to be True and update ValDate to today's date
                self.database.loc[condition, "HumanVal"] = True
                self.database.loc[condition, "ValDate"] = datetime.today().strftime('%Y-%m-%d')
                
                db_annotations = self.database.loc[condition]
                
                self.annotations = [
                    (annotation["x"], annotation["y"])
                    for _, annotation in db_annotations.iterrows()
                    if annotation["x"] != "nan" or annotation["y"] != "nan"
                ]
                
            
            # Draw annotations on the image
            painter = QPainter()
            pixmap = QPixmap.fromImage(image)
            painter.begin(pixmap)
            if self.batch_size == 1 and self.window_x == 1200 and self.window_y == 850:
                pen = QPen(Qt.red, 4)  # Red pen for annotations
            else:    
                pen = QPen(Qt.red, 8)  # Red pen for annotations

            painter.setPen(pen)

            for x, y in self.annotations:
                if x=='nan' or y=='nan':
                    continue
                else:
                    rect = QRect(x, y, self.grid_size, self.grid_size)  
                    painter.drawRect(rect)

            painter.end()
            
            scaled_pixmap_x = self.window_x / self.batch_size * np.sqrt(self.batch_size)
            scaled_pixmap_y = self.window_y / self.batch_size * np.sqrt(self.batch_size)
                
            image_label, filename_label, validation_label = self.image_widgets[i]
            image_label.setPixmap(pixmap.scaled(scaled_pixmap_x, scaled_pixmap_y, Qt.KeepAspectRatio))
            
            filename_label.setText(f"{campaign}/{dut}/{filename}")
            
            # Update the validation label
            if self.current_human_val == False:
                validation_label.setText("NOT VALIDATED")
                validation_label.setStyleSheet("color: red;")
            else:
                validation_label.setText("VALIDATED")
                validation_label.setStyleSheet("color: green;")

        
        # Clear unused labels
        for i in range(len(images), len(self.image_widgets)):
            self.image_widgets[i][0].clear()
            self.image_widgets[i][1].clear()
            self.image_widgets[i][2].clear()
            
    def display_noise_image(self):
        """Display a random noise image in the GUI."""
        noise_image = (np.random.rand(self.image_height, self.image_width) * 255).astype(np.uint8)
        qimage = QImage(noise_image, self.image_width, self.image_height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # Display the noise image on all image labels in the grid
        for image_label, _, _ in self.image_widgets:
            scaled_pixmap_x = self.window_x / self.batch_size * np.sqrt(self.batch_size)
            scaled_pixmap_y = self.window_y / self.batch_size * np.sqrt(self.batch_size)
            image_label.setPixmap(pixmap.scaled(scaled_pixmap_x, scaled_pixmap_y, Qt.KeepAspectRatio))
            
    def mousePressEvent(self, event):
        pos = event.position().toPoint()

        image_label, _, _ = self.image_widgets[0]
        pixmap = image_label.pixmap()
        if pixmap is None:
            return

        label_width = image_label.width()
        label_height = image_label.height()

        x_scale = self.image_width / label_width
        y_scale = self.image_height / label_height

        original_x = int(pos.x() * x_scale)
        original_y = int(pos.y() * y_scale)

        snapped_x = (original_x // self.grid_size) * self.grid_size
        snapped_y = (original_y // self.grid_size) * self.grid_size

        if self.viewing_previous == True:
            components = self.previous_batch[1][0].split(os.sep)
        else:
            components = self.current_batch[1][0].split(os.sep)
            
        campaign = components[-3]
        dut = str(components[-2])
        filename = components[-1][:-4]

        if snapped_x >= 3840 or snapped_y >= 2720:
            return

        if event.button() == Qt.LeftButton:
            if (snapped_x, snapped_y) not in self.annotations:
                self.annotations.append((snapped_x, snapped_y))
                
                # Remove nan row
                placeholder_index = (campaign, dut, filename, "nan")
                self.database = self.database.sort_index() 
                exists = not self.database.loc[[placeholder_index]].empty if placeholder_index in self.database.index else False

                if exists:
                    self.database = self.database.drop(placeholder_index)
                
                new_row_index = (campaign, dut, filename, f"{snapped_x}-{snapped_y}")
                new_row_data = {
                    'HumanVal': True,
                    'x': snapped_x,
                    'y': snapped_y,
                    'ValDate': datetime.today().strftime('%Y-%m-%d'),
                    'Anomaly': True
                }
                new_row_df = pd.DataFrame([new_row_data], index=pd.MultiIndex.from_tuples([new_row_index], names=self.database.index.names))
                self.database = pd.concat([self.database, new_row_df]).sort_index()
                

        elif event.button() == Qt.RightButton:
            if (snapped_x, snapped_y) in self.annotations:
                self.annotations.remove((snapped_x, snapped_y))
                if self.annotations == []:
                    new_row_index = (campaign, dut, filename, "nan")
                    new_row_data = {
                        'HumanVal': True,
                        'x': "nan",
                        'y': "nan",
                        'ValDate': datetime.today().strftime('%Y-%m-%d'),
                        'Anomaly': False
                    }
                    new_row_df = pd.DataFrame([new_row_data], index=pd.MultiIndex.from_tuples([new_row_index], names=self.database.index.names))
                    self.database = pd.concat([self.database, new_row_df])
                
                self.database = self.database.drop((campaign, dut, filename, f"{str(snapped_x)}-{str(snapped_y)}")).sort_index()
                
        if self.viewing_previous == True:
            self.display_batch(self.previous_batch, reset_annotations=False)
        else:
            self.display_batch(self.current_batch, reset_annotations=False)
            
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Space:  # Check if the pressed key is the spacebar
            self.next_button.click() 
        elif event.key() == Qt.Key_Escape:  # Example for Escape key
            self.close()  # Close the application
        else:
            super().keyPressEvent(event)
        
        
        

if __name__ == "__main__":
    # Set up the application
    app = QApplication(sys.argv)
    print("\nStarting GUI...  ")
    
    # manual seed
    manual_seed(314)
    
    BATCH_SIZE = 1
    ONLY_NON_VALIDATED_BOOLEAN = False
    TEST_MODE = False

    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    
    # Load the database
    if TEST_MODE == False:
        # Two different databases for different folder, one for outputs and one for outputs lab pc which is all of the new data
        DB_PATH = os.path.join(ROOT_PATH, "Databases", "updated_validation_DB_CORRECT")
        ROOT_PATH_IMAGES = os.path.join("D:", "CERN", "BackupJuly2024", "outputs")
        # DB_PATH = r"C:\Users\MaxAn\Documents\VScode\CERN\GUIHexaScan\Databases\updated_validation_DB_Lab_PC" 
        # ROOT_PATH_IMAGES = r"D:\CERN\BackupJuly2024\Outputs_Lab_PC" 
    elif TEST_MODE == True:
        print(f"\nStarting in TEST Mode...")
        DB_PATH = os.path.join(ROOT_PATH, "Databases", "10FileNames_validation_DB")
        ROOT_PATH_IMAGES = os.path.join("E:", "CERN", "BackupJuly2024", "TEST_REMOVE")
    
    db = pd.read_pickle(DB_PATH)

    print("\n=============================================== Loaded Database ===============================================\n")
    pd.set_option('display.max.rows', 80)
    print(db)
    # print("\n===============================================================================================================\n")
    
    # Dataset and DataLoader
    # ROOT_PATH_IMAGES = r"E:\CERN\BackupJuly2024\outputs\Preseries_HD_120_MGW_March_2024\600058"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    try:
        dataset = ImageDataset(
                            base_path_images=ROOT_PATH_IMAGES,
                            database=db,
                            transform=transform,
                            only_non_validated=ONLY_NON_VALIDATED_BOOLEAN
                            )
    except Exception as e:
        print(f"\n!! Error loading dataset: {e}", flush=True)
        print(f"\n!! Going into TEST MODE with test dataset in DataTestMode folder in the repository...", flush=True)
        ROOT_PATH_IMAGES = os.path.join(ROOT_PATH, "DataTestMode")
        TEST_MODE = True
        dataset = ImageDataset(
                            base_path_images=ROOT_PATH_IMAGES,
                            database=db,
                            transform=transform,
                            only_non_validated=False
                            )
    
    dataloader = DataLoader(
                        dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=False
                        )
                     
    
    # Create and show the main window
    gui_hexascan = GUI_HexaScan(
                        dataloader=dataloader,
                        database=db,
                        db_path_store=DB_PATH,
                        root_path_images=ROOT_PATH_IMAGES,
                        batch_size=BATCH_SIZE,
                        test_boolean=TEST_MODE
                        )

    gui_hexascan.show()
    
    # Run the application event loop
    sys.exit(app.exec())

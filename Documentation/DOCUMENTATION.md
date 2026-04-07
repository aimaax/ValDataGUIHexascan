# HexaScanAI

## What is the structure of the database? (validation_DB & annotation_DB)

    # Create pandas dataFrame
    columns = ['Campaign', 'DUT', 'FileName', 'bID', 'AI', 'x', 'y', 'ValDate', 'Normal']
    df = pd.DataFrame(all_data, columns=columns)
    
    # Set the multi-index
    df.set_index(['Campaign', 'DUT', 'FileName', 'bID'], inplace=True)
    
    # Write the file to database path (pickle file)
    db_store_path = os.path.join(db_path_abs, "validation_DB")
    df.to_pickle(db_store_path)

This will result in the following structure

    NORMAL VAL (bID == nan and Anomaly == False)
                                                                      HumanVal    x    y     ValDate  Anomaly
    Campaign               DUT    FileName                      bID
    Preseries_December2022 100163 initial_scan_pad13_step30     nan      False  nan  nan  2024-08-10    False
                                initial_scan_pad16_step32       nan      False  nan  nan  2024-08-10    False
                                initial_scan_pad28_step38       nan      False  nan  nan  2024-08-10    False
                                initial_scan_pad16_step41       nan      False  nan  nan  2024-08-10    False
                                initial_scan_pad23_step57       nan      False  nan  nan  2024-08-10    False
    ...                                                                 ...  ...  ...         ...      ...
    ProtoA_October2022     200110 initial_scan_pad171_step357   nan      False  nan  nan  2024-08-09    False
                                initial_scan_pad183_step369     nan      False  nan  nan  2024-08-09    False
                                initial_scan_pad181_step371     nan      False  nan  nan  2024-08-09    False
                                initial_scan_pad180_step373     nan      False  nan  nan  2024-08-09    False
                                initial_scan_pad184_step380     nan      False  nan  nan  2024-08-09    False

    [9374 rows x 5 columns]


    ANOM VAL (all of the anomaly = where b(box)ID is not nan and Anomaly columns == True)

                                                                               HumanVal     x     y     ValDate  Anomaly
    Campaign               DUT      FileName                        bID
    Preseries_December2022 100163   annotated_image_pad6_step8      2240-1760     False  2240  1760  2024-08-10     True
                                                                    2400-1760     False  2400  1760  2024-08-10     True
                                    annotated_image_pad105_step211  3520-640      False  3520   640  2024-08-10     True
                                                                    2880-800      False  2880   800  2024-08-10     True
                                    annotated_image_pad105_step212  1440-320      False  1440   320  2024-08-10     True
    ...                                                                          ...   ...   ...         ...      ...
    ProtoA_October2022     200110   annotated_image_pad157_step330  320-800       False   320   800  2024-08-09     True
                                                                    480-800       False   480   800  2024-08-09     True
                                                                    480-960       False   480   960  2024-08-09     True
                                    annotated_image_pad175_step352  160-1760      False   160  1760  2024-08-09     True
                                    annotated_image_pad185_step379  2880-2560     False  2880  2560  2024-08-09     True

    [3224 rows x 5 columns]


Starting with box coordinates (x = 0, y = 0)

## What is the size of images, i.e. corresponding .npy arrays?

size (width = 3840, height = 2736)

    #number of rows of pixels of the picture
    PICTURESIZE_Y = 2720 # USING THIS
    # PICTURESIZE_Y = 2736 #old picture size

    #number of columns of pixels of the picture
    PICTURESIZE_X = 3840

Take away last 16 values from 2736 to make it even with the annotation boxes

# How many patches are there and what is the size?

Image are divided into 17 rows x 24 cols in 408 patches where each patch size is 160x160. 

Rows: 17 * 160 = 2720
Cols: 24 * 160 = 3840

    REDUCED_DIMENSION = (rows = 17, cols = 24)
    PATCHES = 408
    PATCHSIZE = 160

# Loading stored .npy files have a lot of noise, why? and how to fix?

Why? Still questionable.

**Fix:** Use Gaussian filter on the image tensor with kernel size = (7, 7) and sigma = (2, 2) for best result. It is optimized.

    gaussian_blur = GaussianBlur(kernel_size=(7, 7), sigma=(2, 2))
    image_array = gaussian_blur(image_array.unsqueeze(0)).squeeze(0).squeeze(0)
    image_array = (image_array.numpy() * 255).astype(np.uint8)
    image = QImage(image_array, image_array.shape[1], image_array.shape[0], QImage.Format_Grayscale8)


# Validated databases

**updated_validation_DB_CORRECT** & **updated_validation_DB_Lab_PC** all checked. updated_validation_DB_CORRECT is checked from the stored images in directory ROOT_PATH_IMAGES = r"D:\CERN\BackupJuly2024\outputs" and updated_validation_DB_Lab_PC is checked from the stored images in directory ROOT_PATH_IMAGES = r"D:\CERN\BackupJuly2024\Outputs_Lab_PC". 

Next --> 
1. Delete the rows that have humanVal of false in updated_validation_DB_Lab_PC as the database has some images in the database that does not exist on the stored disk. 
2. Merge the databases together and move all of the campaign to the same folder. Try to run the GUI to see that it is merged correctly.
3. Clean the CERNBox from the previous images and code that are stored there and copy all of the .npy images from the Toshiba 1GB disk as they are correct with the maually checked databases. 
4. Git clone the repo to CERNBox, add a Gaussian filter (see optimized Gaussian kernel size and sigma values above) before the image is processed to the AI pipeline. 
5. Fingers cross that it will work :)

Worked :)
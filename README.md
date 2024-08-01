# EEG Data Processing and Classification

I was trying to emulate the results of the article "[One-dimensional convolutional neural network-based identification of sleep disorders using electroencephalogram signals](https://www.sciencedirect.com/science/article/abs/pii/B978032396129500010X)

The article needed to explain more of the pre-processing steps. It was very brief. I manage to make it work, but my accuracies for both models were lower. I used 12 epochs.


This repository provides tools and code for processing and classifying EEG data using convolutional neural networks (CNNs) in PyTorch. The process involves several key steps, from data extraction and preprocessing to model training and evaluation. 

## Table of Contents

1. [Data Folder Setup](#data-folder-setup)
1. [Dependencies](#dependencies)
2. [Data Extraction and Preprocessing](#data-extraction-and-preprocessing)
3. [Dataset Creation](#dataset-creation)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)

## Data Folder Setup

To run the code and reproduce the experiments, you need to organize your data folder in a specific structure. Below is the recommended setup for the data folder:

1. **Data Folder Structure**:
    - Create a main data folder named `Data`.
    - Inside the `Data` folder, create a subfolder named `Sleep`.
    - Inside the `Sleep` folder, create another subfolder named `sleep_cassette`.

    The folder structure should look like this:
    ```
    Data/
    └── Sleep/
        └── sleep_cassette/
            ├── SC4001E0-PSG.edf
            ├── SC4001EC-Hypnogram.edf
            ├── SC4002E0-PSG.edf
            ├── SC4002EC-Hypnogram.edf
            ├── ...
    ```

2. **File Naming Convention**:
    - The EEG signal files should have the format `SCxxxxE0-PSG.edf`, where `xxxx` is the subject ID.
    - The annotation files (hypnograms) should have the format `SCxxxxEC-Hypnogram.edf`, where `xxxx` is the subject ID.

3. **Data Folder Path**:
    - In the code, set the `data_folder` variable to the path of the `sleep_cassette` folder.
    ```python
    data_folder = "/content/drive/MyDrive/Data/Sleep/sleep_cassette/"
    ```

4. **Example Files**:
    - Ensure that the `sleep_cassette` folder contains the EEG signal files and corresponding annotation files for each subject.
    - Example files:
        - `SC4001E0-PSG.edf`: EEG signal file for subject SC4001.
        - `SC4001EC-Hypnogram.edf`: Annotation file for subject SC4001.
        - `SC4002E0-PSG.edf`: EEG signal file for subject SC4002.
        - `SC4002EC-Hypnogram.edf`: Annotation file for subject SC4002.

By organizing your data folder in this structure, the code will be able to locate and load the EEG signal and annotation files correctly. Make sure to update the `data_folder` variable in the code to match the path where your data is stored.

Feel free to include this in your `README.md` file. If you have any more questions or need further details, let me know! 😊



## Dependencies

Ensure the following Python packages are installed:

- `numpy`
- `mne`
- `torch`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pandas`

## Install these using pip:

#### pip install numpy mne torch matplotlib seaborn scikit-learn pandas 

## Data Extraction and Preprocessing

### Extracting EDF Data

The function `load_data_refactored` and  `extract_epochs` handle the extraction of EEG data from EDF files. It uses the MNE library to read raw data files and annotate the events.

### Preprocessing Steps

1. **Loading Data**: Load the raw EDF files using MNE.
2. **Annotating Events**: Annotate events using descriptions provided in the EDF files.
3. **Filtering**: Apply band-pass filters to the data.
4. **Epoching**: Segment the data into 30-second and 60-second epochs.

## Dataset Creation

### 1. 30-Second Segments Dataset

**Description**:
- This dataset consists of 30-second epochs extracted from the raw EEG data.
- Each epoch represents a 30-second segment of EEG signals.
- The dataset includes annotations for different sleep stages, such as Wake (W), N1, N2, Slow Wave Sleep (SW), and REM (R).
- The data is normalized using the `scale` function to ensure consistent input for the deep learning models.

**Processing Steps**:
1. **Data Loading**: Load the raw EEG data using MNE.
2. **Channel Selection**: Select relevant EEG channels (e.g., Fpz-Cz).
3. **Epoch Extraction**: Extract 30-second epochs from the raw data.
4. **Normalization**: Normalize the data using the `scale` function.
5. **Dataset Creation**: Create a custom `Dataset` class for 30-second segments.

### 2. 60-Second Segments Dataset

**Description**:
- This dataset consists of 60-second epochs created by concatenating consecutive 30-second epochs.
- Each epoch represents a 60-second segment of EEG signals.
- The dataset includes annotations for different sleep stages, similar to the 30-second segments dataset.
- The data is normalized using the `scale2` function to ensure consistent input for the deep learning models.

**Processing Steps**:
1. **Data Loading**: Load the raw EEG data using MNE.
2. **Channel Selection**: Select relevant EEG channels (e.g., Fpz-Cz).
3. **Epoch Extraction**: Extract 30-second epochs from the raw data.
4. **Concatenation**: Create consecutive 30-second epochs and concatenate them to form 60-second segments.
5. **Filtering**: Filter the concatenated segments to keep only those with matching annotations.
6. **Normalization**: Normalize the data using the `scale2` function.
7. **Dataset Creation**: Create a custom `Dataset` class for 60-second segments.

These datasets are used to train and evaluate the deep learning models for sleep stage classification. The 30-second segments dataset provides a finer temporal resolution, while the 60-second segments dataset offers a longer context for each epoch.


## Model Architecture

### Models

#### CustomModel

**Description**:
- The `CustomModel` is a deep learning model designed for sleep stage classification using 1D EEG data.
- It consists of multiple convolutional layers, pooling layers, and fully connected layers.
- The model uses ReLU activation functions and dropout layers to prevent overfitting.

**Architecture**:
- **Conv1**: 1D Convolutional layer with 64 filters, kernel size of 5, and stride of 3.
- **Conv2**: 1D Convolutional layer with 128 filters, kernel size of 5, and stride of 1.
- **Pool1**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Dropout1**: Dropout layer with a dropout rate of 0.4.
- **Conv3**: 1D Convolutional layer with 128 filters, kernel size of 13, and stride of 1.
- **Conv4**: 1D Convolutional layer with 256 filters, kernel size of 7, and stride of 1.
- **Pool2**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Conv5**: 1D Convolutional layer with 256 filters, kernel size of 7, and stride of 1.
- **Conv6**: 1D Convolutional layer with 64 filters, kernel size of 4, and stride of 1.
- **Pool3**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Conv7**: 1D Convolutional layer with 64 filters, kernel size of 3, and stride of 1.
- **Conv8**: 1D Convolutional layer with 64 filters, kernel size of 6, and stride of 1.
- **Pool4**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Conv9**: 1D Convolutional layer with 8 filters, kernel size of 5, and stride of 1.
- **Conv10**: 1D Convolutional layer with 8 filters, kernel size of 2, and stride of 1.
- **Pool5**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Flatten**: Flatten layer to convert the 3D tensor to 1D.
- **FC1**: Fully connected layer with 64 units.
- **Dropout2**: Dropout layer with a dropout rate of 0.4.
- **FC2**: Fully connected layer with `num_classes` units (output layer).

**Forward Pass**:
- The input passes through the convolutional layers with ReLU activation and pooling layers.
- The output is flattened and passed through fully connected layers with ReLU activation and dropout layers.
- The final output is obtained from the last fully connected layer.

#### RandomModel

**Description**:
- The `RandomModel` is another deep learning model designed for sleep stage classification using 1D EEG data.
- It has a different architecture compared to the `CustomModel`, with varying convolutional layers, pooling layers, and fully connected layers.
- The model uses ReLU activation functions and dropout layers to prevent overfitting.

**Architecture**:
- **Conv1**: 1D Convolutional layer with 128 filters, kernel size of 3, and stride of 3.
- **Conv2**: 1D Convolutional layer with 152 filters, kernel size of 3, and stride of 1.
- **Pool1**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Dropout1**: Dropout layer with a dropout rate of 0.6.
- **Conv3**: 1D Convolutional layer with 232 filters, kernel size of 9, and stride of 1.
- **Conv4**: 1D Convolutional layer with 56 filters, kernel size of 9, and stride of 1.
- **Pool2**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Conv5**: 1D Convolutional layer with 240 filters, kernel size of 5, and stride of 1.
- **Conv6**: 1D Convolutional layer with 64 filters, kernel size of 4, and stride of 1.
- **Pool3**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Conv7**: 1D Convolutional layer with 48 filters, kernel size of 11, and stride of 1.
- **Conv8**: 1D Convolutional layer with 200 filters, kernel size of 9, and stride of 1.
- **Pool4**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Pool5**: Max Pooling layer with kernel size of 2 and stride of 2.
- **Flatten**: Flatten layer to convert the 3D tensor to 1D.
- **FC1**: Fully connected layer with 88 units.
- **Dropout2**: Dropout layer with a dropout rate of 0.6.
- **FC2**: Fully connected layer with `num_classes` units (output layer).

**Forward Pass**:
- The input passes through the convolutional layers with ReLU activation and pooling layers.
- The output is flattened and passed through fully connected layers with ReLU activation and dropout layers.
- The final output is obtained from the last fully connected layer.


## Training and Evaluation

### Training, Evaluation, and Testing

#### Training Steps

1. **Data Preparation**:
    - Split the data into training, validation, and test sets.
    - Create data loaders for each set using the `DataLoader` class from PyTorch.

2. **Model Initialization**:
    - Initialize the model (`CustomModel` or `RandomModel`) with the appropriate input size and number of classes.
    - Define the loss function (`CrossEntropyLoss`) and the optimizer (`Adam`).

3. **Training Loop**:
    - For each epoch:
        - Set the model to training mode.
        - Iterate over the training data:
            - Load the input data and labels.
            - Perform a forward pass through the model.
            - Compute the loss.
            - Perform a backward pass to compute gradients.
            - Update the model parameters using the optimizer.
        - Compute the training accuracy.
        - Validate the model on the validation set and compute the validation accuracy.
        - Implement early stopping based on validation accuracy to prevent overfitting.

4. **Early Stopping**:
    - Monitor the validation accuracy during training.
    - If the validation accuracy does not improve for a specified number of epochs (patience), stop the training early.

#### Evaluation and Testing

1. **Validation**:
    - After each epoch, evaluate the model on the validation set.
    - Compute the validation loss and accuracy.
    - Use the validation accuracy to decide whether to continue training or stop early.

2. **Testing**:
    - After training is complete, evaluate the model on the test set.
    - Compute the test accuracy to assess the model's performance on unseen data.

3. **Confusion Matrix**:
    - Plot the confusion matrix to visualize the model's performance across different sleep stages.
    - The confusion matrix shows the true labels vs. the predicted labels, providing insights into the model's classification accuracy for each sleep stage.

#### Example Code for Training and Evaluation

```python
# Initialize the model, criterion, and optimizer
input_size = train_ds_30s[0][0].shape[0]
num_classes = len(EVENT_ID)
model = CustomModel(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_30s, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_ds_30s, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_ds_30s, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds_30s, batch_size=batch_size, shuffle=False)

# Initialize the trainer and start training
trainer = Trainer(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=50, patience=5)
trainer.train()

# Evaluate on the test set
trainer.test()

# Plot confusion matrix
trainer.plot_confusion_matrix()



# Image Classification Tutorial with Knowledge Distillation

   This repository contains a Jupyter notebook (`part3.ipynb`) that guides you through the process of training image classification models using PyTorch, with a focus on knowledge distillation. The tutorial covers preprocessing a dataset, training a teacher model (ResNet34), training a student model (ResNet18) using knowledge distillation, and making predictions on a test set.

   ## Prerequisites

   To run the notebook, ensure you have the following installed:
   - **Anaconda** (recommended for managing Python environments)
   - Python 3.10
   - Jupyter Notebook or JupyterLab
   - Required Python libraries (specified in `MSA.yaml`)

   ## Setting Up the Environment with Anaconda

   We recommend using **Anaconda** to create a Python environment with all the required dependencies. Follow these steps to set up the environment:

   1. **Install Anaconda**:
      - Download and install Anaconda from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) if you haven't already.
      - Follow the installation instructions for your operating system.

   2. **Create the Base Environment**:
      - Open the Anaconda Prompt or a terminal and navigate to the directory containing the `MSA.yaml` file.
      - Create the environment using the provided `MSA.yaml` file:
        ```bash
        conda env create -f MSA.yaml
        ```
      - This will create an environment named `MSA` with most dependencies, excluding PyTorch and torchvision, which will be installed based on your device's CUDA version.

   3. **Install PyTorch and torchvision**:
      - Activate the `MSA` environment:
        ```bash
        conda activate MSA
        ```
      - Check your device's CUDA version by running:
        ```bash
        nvidia-smi
        ```
        If no GPU is available, use the CPU version.
      - Install `torch` and `torchvision` using the appropriate command from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). Examples:
        - **CUDA 12.1**:
          ```bash
          pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
          ```
        - **CUDA 11.8**:
          ```bash
          pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
          ```
        - **CPU-only**:
          ```bash
          pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
          ```
      - Verify the installation:
        ```bash
        python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
        ```

   4. **Activate the Environment**:
      - Ensure the `MSA` environment is activated before running the notebook:
        ```bash
        conda activate MSA
        ```

   ## Folder Structure

   The project expects the following folder structure:

   ```
   Kaggle/
   ├── dataset_split/
   │   ├── train/
   │   │   ├── african_elephant (780 images)
   │   │   ├── airliner (780 images)
   │   │   └── ... (8 more folders)
   │   ├── val/
   │   │   ├── african_elephant (260 images)
   │   │   ├── airliner (260 images)
   │   │   └── ... (8 more folders)
   │   └── test (2600 images)
   ├── data_preprocessed/
   │   ├── labels_train.pt
   │   ├── labels_val.pt
   │   ├── tensor_test.pt
   │   ├── tensor_train.pt
   │   └── tensor_val.pt
   ├── models/
   │   ├── resnet18_checkpoint.pkl
   │   └── resnet34_checkpoint.pkl
   ├── src/
   │   └── part3.ipynb
   ├── src_dataSplit/
   │   └── data_split.ipynb
   ├── MSA.yaml
   ├── README.md
   └── IEEE_Report_Template.docx
   ```

   ## Tutorial Steps

   The notebook is divided into four main steps:

   1. **Preprocessing**:
      
      - Load images and labels from the dataset.
      - Resize images to 224x224 pixels.
      - Convert images to float32 and normalize to [0, 1].
      - Transform data to NCHW format and create PyTorch tensors.
      - Standardize data using training set statistics.
      - Save preprocessed data as `.pt` files.
      - Load preprocessed data for subsequent steps.
      
   2. **Train the Teacher Model (ResNet34)**:
      - Set hyperparameters (e.g., learning rate, batch size, epochs).
      - Initialize a pretrained ResNet34 model and modify its output layer.
      - Define the optimizer and loss function.
      - Train the model and save the best parameters to `resnet34_checkpoint.pkl`.
      - Plot training and validation loss/accuracy curves.

   3. **Train the Student Model (ResNet18 with Knowledge Distillation)**:
      - Set hyperparameters, including temperature and loss ratio for distillation.
      - Initialize a non-pretrained ResNet18 model.
      - Load the trained ResNet34 model as the teacher.
      - Define the optimizer and knowledge distillation loss function.
      - Train the ResNet18 model and save the best parameters to `resnet18_checkpoint.pkl`.
      - Plot training and validation loss/accuracy curves.

   4. **Predict Labels for the Test Set**:
      - Load the trained ResNet18 model.
      - Perform predictions on the test set images.
      - Save predictions to a `submission.csv` file in the format:
        ```
        file_name,label
        0.jpg,sunglasses
        ...
        ```

   ## Running the Notebook

   1. Clone this repository or download the notebook and dataset.
   2. Ensure the dataset is organized as described in the folder structure.
   3. Activate the `MSA` environment:
      ```bash
      conda activate MSA
      ```
   4. Open the `part3.ipynb` notebook in Jupyter:
      ```bash
      jupyter notebook src/part3.ipynb
      ```
   5. Follow the instructions in the notebook, completing the `TODO` sections with the appropriate code.
   6. Run each cell sequentially to preprocess data, train models, and generate predictions.

   ## Notes

   - The dataset should contain 10 classes, with 780 training and 260 validation images per class, and 2600 test images.
   - Ensure GPU availability for faster training (the notebook automatically detects CUDA if available, falling back to CPU if not).
   - Knowledge distillation is used to transfer knowledge from the larger ResNet34 model to the smaller ResNet18 model, improving performance with fewer parameters.
   - The final `submission.csv` file is used for evaluating the model's performance on the test set.

   ## Expected Output

   - Plots of loss and accuracy curves for both ResNet34 and ResNet18.
   - A `submission.csv` file with test set predictions, which should be uploaded to Kaggle for accuracy test.
   - **A comprehensive technical report**

   ## Submission

   Each student must submit the following:

   ### 1. Notebook Submission (5 marks)
   - A fully completed and followed good programming practices Jupyter notebook `part3.ipynb`, including:
      - All code implementations (preprocessing, training, distillation, predictions)
      - Plots of training/validation performance

   ### 2. Final predictions in `submission.csv` (5 marks)
   - This is a `CSV` formatted table containing two columns:
      - The first column, `file_name`, indicates the name of the image file (e.g., "0.jpg", "1.jpg").
      - The second column, label, represents the corresponding class label for the image (e.g., "sunglasses" "convertible_car").
     - For example:
        ```
        file_name,label
        0.jpg,sunglasses
        1.jpg,convertible_car
        ```

   ### 3. Report Submission (50 marks)
- The report must follow **IEEE format**, including:
  - **Title, Abstract, Sections** properly organized
  - **Figures and Tables** labeled and referenced
  - **Reference** formatted according to IEEE citation standards
  - **Technical writing** with correct grammar and clear explanation

#### Report Contents
- **Introduction**: 

  - Description: sets the stage for the report by introducing the problem and objectives
  - Criteria:
    - Clearly defines the problem
    - Provides motivation and context for the work
    - Outlines the structure of the report

- **Literature Review**:

  - Description: Demonstrates understanding of existing work related to knowledge distillation and image classification
  - Criteria: 

    - Summarizes at least 6 relevant papers or resources
    - Identifies gaps or limitations in existing work

    - Relates the current assignment to the literature

- **Methodology**: 

  - Description: Details the technical approach, including preprocessing, model training, and distillation process
  - Criteria: 

    - Describes data preprocessing steps

    - Explains the model architecture and training process

    - Explains the knowledge distillation

    - Justifies the choice of hyperparameters

- **Results**:

  - Description: Presents the outcomes of the project, focusing on model performance
  - Criteria:

    - Presents accuracy and other relevant metrics

    - Includes appropriate visualizations

    - Compares results with the teacher model

    - Compares computational efficiency

- **Discussion**:

  - Description: Interprets the results and reflects on the understanding of the process and outcomes
  - Criteria:

    - Interprets the results in the context of the objectives

    - Discusses the implications of the findings

    - Addresses limitations and potential improvements

    - Reflects on the learning experience and understanding of knowledge distillation

- **Conclusion**:

  - Description: Summarizes the work and suggests future directions

  - Criteria:

    - Summarizes the key findings
    - Suggests future research directions

    - Reflects on the overall contribution and significance of the work




   ## Troubleshooting

   - **Environment Issues**:
     - If the environment creation fails, ensure the `MSA.yaml` file is in the correct directory and that you have an active internet connection.
     - Verify that Anaconda is up to date:
       ```bash
       conda update -n base conda
       ```
     - If PyTorch installation fails, ensure the `--index-url` matches the desired CUDA version or CPU setup.
   - **Memory Issues**:
     - If you encounter memory issues during training, reduce the batch size in the hyperparameters.
   - **Dataset Paths**:
     - Verify that the dataset paths in the notebook match your local folder structure.
   - **CUDA Compatibility**:
     - If PyTorch reports CUDA errors, ensure the installed PyTorch version matches the device's CUDA version. Reinstall with the correct command from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
   - **Troubleshooting sessions**:
     - If you encounter further issues, please attend the troubleshooting sessions held on Discord.

   For additional help, refer to the [PyTorch documentation](https://pytorch.org/docs/stable/index.html), [Anaconda documentation](https://docs.anaconda.com/), or relevant knowledge distillation tutorials.
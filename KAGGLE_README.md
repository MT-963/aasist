# Running AASIST on Kaggle

This guide explains how to run the AASIST project on Kaggle.

## Setup Instructions

1. **Create a New Kaggle Notebook**
   - Create a new notebook in Kaggle
   - Make sure to enable GPU accelerator in the notebook settings

2. **Upload the Project Files**
   - Upload all project files to your Kaggle notebook
   - Required files:
     - `main.py`
     - `data_utils.py`
     - `utils.py`
     - `evaluation.py`
     - `adversarial_attack.py`
     - `requirements.txt`
     - `config/` directory with configuration files
     - `models/` directory with model definitions

3. **Install Dependencies**
   ```python
   !pip install -r requirements.txt
   ```

4. **Download and Prepare the Dataset**
   ```python
   # Create necessary directories
   !mkdir -p /kaggle/working/flac
   
   # Download the LA dataset (you'll need to provide the download link)
   # For example:
   !wget -O /kaggle/working/LA.zip <your_dataset_download_link>
   
   # Extract the dataset
   !unzip /kaggle/working/LA.zip -d /kaggle/working/
   
   # Move flac files to the correct location
   !mv /kaggle/working/ASVspoof2019_LA_train/flac/* /kaggle/working/flac/
   !mv /kaggle/working/ASVspoof2019_LA_dev/flac/* /kaggle/working/flac/
   !mv /kaggle/working/ASVspoof2019_LA_eval/flac/* /kaggle/working/flac/
   
   # Copy protocol files
   !cp /kaggle/working/ASVspoof2019_LA_cm_protocols/*.txt /kaggle/working/
   ```

5. **Run Training or Evaluation**
   ```python
   # For training:
   !python main.py --config ./config/AASIST.conf
   
   # For evaluation:
   !python main.py --eval --config ./config/AASIST.conf
   ```

## Important Notes

1. **Memory Management**
   - The code has been modified to handle Kaggle's GPU memory constraints
   - Batch sizes are automatically adjusted based on available GPU memory
   - If you encounter memory issues, you can modify the batch size in the config file

2. **File Structure**
   - All data should be in `/kaggle/working/`
   - Audio files should be in `/kaggle/working/flac/`
   - Protocol files should be in `/kaggle/working/`
   - Model outputs will be saved in `/kaggle/working/exp_result/`

3. **Troubleshooting**
   - If you get file not found errors, verify that:
     - All flac files are in `/kaggle/working/flac/`
     - Protocol files are in `/kaggle/working/`
     - File permissions are correct
   - If you get memory errors:
     - Reduce the batch size in the config file
     - Clear GPU memory using `torch.cuda.empty_cache()`
     - Restart the notebook if needed

4. **Saving Results**
   - All results and models will be saved in `/kaggle/working/`
   - Make sure to download important files before the session ends
   - You can use Kaggle's dataset feature to save your trained models

## Example Notebook Structure

```python
# Cell 1: Install dependencies
!pip install -r requirements.txt

# Cell 2: Setup directories and download data
!mkdir -p /kaggle/working/flac
# ... download and extract dataset ...

# Cell 3: Run training or evaluation
!python main.py --config ./config/AASIST.conf

# Cell 4: Save results (if needed)
!cp -r /kaggle/working/exp_result /kaggle/working/models /kaggle/working/results
```

## Performance Considerations

1. **GPU Memory**
   - The code automatically adjusts batch sizes for Kaggle's GPU
   - Default settings should work on most Kaggle GPUs
   - If you need to use a smaller model, try AASIST-L instead of AASIST

2. **Training Time**
   - Full training may take several hours
   - Consider using a smaller number of epochs for testing
   - You can modify `num_epochs` in the config file

3. **Evaluation**
   - Evaluation is faster than training
   - You can use pre-trained models for quick testing
   - Results will be saved in `/kaggle/working/exp_result/` 
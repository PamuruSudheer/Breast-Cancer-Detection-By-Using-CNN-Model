# Cancer Detection AI Server

A Flask-based web application for breast cancer detection using a CNN model trained on the BUSI (Breast Ultrasound Images) dataset.

## Project Structure

```
4th_year_server/
├── app.py                    # Flask server application
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── templates/
│   └── index.html          # Web interface
└── static/
    ├── style.css           # Styling
    └── script.js           # Client-side logic
```

## Features

- **Web Interface**: User-friendly drag-and-drop image upload
- **Real-time Predictions**: Instant classification with confidence scores
- **REST API**: JSON endpoints for programmatic access
- **Model Visualization**: Class probabilities displayed with progress bars
- **Health Checks**: Server and model status monitoring

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Sufficient disk space for model (~100MB)

## Installation

### 1. Install Dependencies

```bash
# Navigate to the server directory
cd c:\Users\sudhe\4th_year_server

# Install Python packages
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Run the training script
python train_model.py
```

This will:
- Load images from the BUSI dataset
- Train the CNN model for 15 epochs
- Save the model to `model/cnn_model.keras`
- Generate training results visualization

**Expected Output:**
```
======================================================================
CANCER DETECTION MODEL TRAINING
======================================================================

[1/4] Loading dataset...
✓ Dataset loaded successfully!
  - Total images: 782
  - Benign: 438
  - Malignant: 210
  - Normal: 134

[2/4] Preprocessing data...
✓ Data preprocessing completed!
  - Training samples: 625 (80%)
  - Test samples: 157 (20%)

[3/4] Building and training CNN model...
✓ Model training completed!

[4/4] Evaluating model...
✓ Model evaluation completed!
  - Accuracy:  XX.XX%
  - Precision: XX.XX%
  - Recall:    XX.XX%
  - F1-Score:  XX.XX%

======================================================================
✓ TRAINING COMPLETED SUCCESSFULLY!
======================================================================
```

### 3. Run the Server

```bash
# Start the Flask development server
python app.py
```

**Expected Output:**
```
======================================================================
CANCER DETECTION SERVER
======================================================================

Loading trained model...
✓ Model loaded successfully!

✓ Server is ready!

Available endpoints:
  GET  /                 - Web interface
  GET  /api/health       - Health check
  GET  /api/info         - Server information
  POST /api/predict      - Image prediction

======================================================================
Server running on: http://localhost:5000
======================================================================
```

## Usage

### Web Interface

1. Open your browser and navigate to: `http://localhost:5000`
2. Click "Select Image" or drag-and-drop an image
3. View the prediction results with confidence scores

### API Endpoints

#### Health Check
```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "message": "Cancer Detection Server is running"
}
```

#### Server Info
```bash
curl http://localhost:5000/api/info
```

**Response:**
```json
{
  "name": "Cancer Detection API",
  "version": "1.0.0",
  "description": "Breast cancer detection using CNN model",
  "classes": {
    "0": "Benign",
    "1": "Malignant",
    "2": "Normal"
  },
  "model_status": "loaded"
}
```

#### Image Prediction
```bash
curl -X POST -F "image=@path/to/image.png" http://localhost:5000/api/predict
```

**Response:**
```json
{
  "success": true,
  "filename": "image.png",
  "prediction": "Benign",
  "confidence": 95.42,
  "probabilities": {
    "benign": 95.42,
    "malignant": 3.21,
    "normal": 1.37
  }
}
```

## Model Architecture

```
Input (32x32x3)
    ↓
Conv2D (32 filters, 3x3)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (32 filters, 3x3)
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
RepeatVector (2)
    ↓
LSTM (32 units)
    ↓
Dense (256 units, ReLU)
    ↓
Dense (3 units, Softmax)
    ↓
Output (Benign, Malignant, Normal)
```

## Classes

- **Benign (0)**: Non-cancerous growths
- **Malignant (1)**: Cancerous tissue
- **Normal (2)**: Healthy tissue

## Dataset

The model is trained on the BUSI (Breast Ultrasound Images) dataset:
- Location: `C:\Users\sudhe\OneDrive\Desktop\main_project\cancerdetection\Dataset_BUSI_with_GT`
- Total Images: ~782
- Classes: 3
- Image Size: 32x32 (resized for model input)

## Performance Metrics

After training, the model provides:
- **Accuracy**: Classification accuracy on test set
- **Precision**: True positive rate for each class
- **Recall**: Sensitivity of the model
- **F1-Score**: Harmonic mean of precision and recall

## Troubleshooting

### Model Not Loading
**Error:** `Model file not found at model/cnn_model.keras`

**Solution:**
```bash
# Train the model first
python train_model.py
```

### Port Already in Use
**Error:** `Address already in use`

**Solution:** Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use port 5001 instead
```

### Insufficient Memory
**Error:** CUDA/Memory errors during training

**Solution:** Reduce batch size in `train_model.py`:
```python
batch_size=16  # Instead of 32
```

### Dependencies Installation Failed
**Solution:** Install packages individually:
```bash
pip install tensorflow opencv-python flask scikit-learn
```

## Configuration

### Model Parameters
Edit `train_model.py`:
- `IMAGE_SIZE`: Input image dimension (default: 32)
- `epochs`: Training epochs (default: 15)
- `batch_size`: Batch size (default: 32)

### Server Parameters
Edit `app.py`:
- `UPLOAD_FOLDER`: Temporary upload directory
- `MAX_FILE_SIZE`: Maximum upload size (default: 50MB)
- `port`: Server port (default: 5000)

## Performance Tips

1. **GPU Acceleration**: Install CUDA and cuDNN for faster training
2. **Memory Optimization**: Use smaller batch sizes for limited RAM
3. **Data Augmentation**: Add image augmentation for better generalization
4. **Model Checkpointing**: Model automatically saves best weights

## Security Notes

⚠️ **Important for Production:**
- Set `debug=False` in production
- Use `HTTPS` for real deployments
- Implement authentication for API access
- Validate all user inputs
- Use proper error handling

## Disclaimer

⚠️ **Medical Disclaimer:**
This tool is for educational and research purposes only. It should NOT be used for actual medical diagnosis without professional medical consultation. Always consult with qualified healthcare professionals for medical decisions.

## License

This project is created for educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review server logs
3. Ensure all dependencies are installed
4. Verify dataset path is correct

## Future Improvements

- [ ] Model ensembling
- [ ] Transfer learning with pre-trained models
- [ ] Batch prediction API
- [ ] Model explanation (LIME/SHAP)
- [ ] User authentication
- [ ] Prediction history logging
- [ ] Multi-model comparison
- [ ] Web UI improvements with charts

---

**Created:** January 2026
**Version:** 1.0.0

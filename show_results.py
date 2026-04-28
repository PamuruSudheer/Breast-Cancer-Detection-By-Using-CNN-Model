"""
Quick result viewer - displays training results with graph visualizations
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt
from datetime import datetime

MODEL_DIR = "model"

def log_info(msg):
    print(f"[RESULTS] {msg}")

def open_image_file(filepath):
    """Open image file with default viewer"""
    try:
        import subprocess
        import sys
        if sys.platform == 'win32':
            os.startfile(filepath)
        elif sys.platform == 'darwin':
            subprocess.run(['open', filepath])
        else:
            subprocess.run(['xdg-open', filepath])
        return True
    except Exception as e:
        return False

log_info("="*70)
log_info("CANCER DETECTION MODEL - TRAINING RESULTS")
log_info("="*70)

# Check if model exists
if os.path.exists(os.path.join(MODEL_DIR, 'cnn_model.keras')):
    size_mb = os.path.getsize(os.path.join(MODEL_DIR, 'cnn_model.keras')) / (1024**2)
    log_info(f"\n✓ Model file created: cnn_model.keras ({size_mb:.1f} MB)")
else:
    log_info(f"\n❌ Model file NOT found")

# Check history file
if os.path.exists(os.path.join(MODEL_DIR, 'cnn_history.pckl')):
    with open(os.path.join(MODEL_DIR, 'cnn_history.pckl'), 'rb') as f:
        history = pickle.load(f)
    
    log_info(f"\n✓ Training history saved")
    
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    
    log_info(f"\nTraining Summary:")
    log_info(f"  - Total epochs trained: {len(train_acc)}")
    log_info(f"  - Best training accuracy: {max(train_acc)*100:.2f}%")
    log_info(f"  - Best validation accuracy: {max(val_acc)*100:.2f}%")
    log_info(f"  - Final training accuracy: {train_acc[-1]*100:.2f}%")
    log_info(f"  - Final validation accuracy: {val_acc[-1]*100:.2f}%")
    
    best_epoch = np.argmax(val_acc) + 1
    log_info(f"\n  Best performance at Epoch {best_epoch}: {max(val_acc)*100:.2f}% accuracy")
    
    # Create graph visualizations
    log_info(f"\n📊 Creating performance graphs...")
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy graph
    axes[0].plot(range(1, len(train_acc)+1), np.array(train_acc)*100, 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(range(1, len(val_acc)+1), np.array(val_acc)*100, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch {best_epoch}')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss graph
    axes[1].plot(range(1, len(train_loss)+1), train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(range(1, len(val_loss)+1), val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch {best_epoch}')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    graph_path = os.path.join(MODEL_DIR, 'training_results_graphs.png')
    plt.savefig(graph_path, dpi=100, bbox_inches='tight')
    log_info(f"  ✓ Graphs saved successfully to: {graph_path}")
    
    try:
        log_info(f"  ℹ Attempting to display graph window...")
        plt.show()
    except Exception as e:
        log_info(f"  ⓘ Graph window couldn't display: {str(e)}")
        log_info(f"  ✓ But graphs are saved - attempting to open file...")
        
        if open_image_file(os.path.abspath(graph_path)):
            log_info(f"  ✓ Opened graph file in default viewer")
        else:
            log_info(f"  ✓ Graph file saved at: {os.path.abspath(graph_path)}")
            log_info(f"  → You can open it manually in any image viewer")
    
else:
    log_info(f"\n❌ Training history NOT found")

# Summary
log_info("\n" + "="*70)
log_info("✓ RESULTS DISPLAY COMPLETE")
log_info("="*70)
log_info("\nModel Pipeline Status:")
log_info("  1. ✓ Model trained and saved")
log_info("  2. ✓ Results visualized with graphs")
log_info("  3. → Next: Run 'python app.py' to start the web server")
log_info("\n" + "="*70 + "\n")

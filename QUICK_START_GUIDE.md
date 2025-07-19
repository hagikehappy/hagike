# Hagike Toolkit - Quick Start Guide

## Installation & Setup

```bash
# Clone and install
git clone https://github.com/hagikehappy/hagike.git
cd hagike
pip install -r requirements.txt
pip install -e .
```

## 5-Minute Quick Start

### 1. Basic Utilities

```python
# Message system with colors
from hagike.utils import add_msg, MsgLevel

add_msg(MsgLevel.Run.value, "Starting application")
add_msg(MsgLevel.Warning.value, "Memory usage high")
add_msg(MsgLevel.Error.value, "File not found")
```

### 2. Configuration Management

```python
# JSON config handling
from hagike.utils import read_json, write_json

# Save config
config = {
    "model": {"learning_rate": 0.001, "batch_size": 32},
    "data": {"path": "/data", "augment": True}
}
write_json("config.json", config)

# Load config
loaded_config = read_json("config.json")
print(f"Learning rate: {loaded_config['model']['learning_rate']}")
```

### 3. Advanced Enums

```python
from hagike.utils import SuperEnum, advanced_enum

@advanced_enum()
class Config(SuperEnum):
    learning_rate = 0.001
    batch_size = 32
    
    class Model(SuperEnum):
        architecture = "resnet50"
        pretrained = True

# Usage
config_dict = Config.dict_()
lr = Config.get_value_(Config.learning_rate)
Config.tree_(is_value=True)  # Print structure
```

### 4. Computer Vision

```python
from hagike.basics.cv import *

# Load and enhance image
image = ImStd.load("photo.jpg")
enhanced = im_log(image)  # Enhance dark regions

# Apply filters
smoothed = apply_smoothing_filter(enhanced, ImSmooth.gaussian.value)
sharpened = apply_sharpening_filter(smoothed, ImSharp.laplacian.value)

# Histogram operations
hist = convert_histogram(image)
equalized_img, _, _ = flatten_histogram(image)
draw_histogram(hist, save="hist.png")
```

### 5. Model Framework

```python
from hagike.models.temp import ModuleNode, ModuleMode
import torch.nn as nn

# Simple model
class MyModel(ModuleNode):
    def __init__(self):
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        super().__init__(model)

# Usage
model = MyModel()
model.to(device="cuda", mode=ModuleMode.train.value)
model.save_weights("model.pth")
```

### 6. Logging System

```python
from hagike.log import logger_g, LogTemp

# Structured logging
log_entry = LogTemp(
    level=1,
    src="main",
    msg="Processing started",
    event_main=100
)

logger_g.add_log(log_entry)
logger_g.print_msg("Simple message")
```

## Common Patterns

### Configuration Pattern

```python
@advanced_enum()
class ExperimentConfig(SuperEnum):
    # Define experiment parameters
    seed = 42
    device = "cuda"
    
    class Data(SuperEnum):
        batch_size = 32
        num_workers = 4
        train_split = 0.8
    
    class Model(SuperEnum):
        hidden_size = 256
        dropout = 0.1
        
    class Training(SuperEnum):
        epochs = 100
        learning_rate = 0.001
        weight_decay = 1e-4

# Use configuration
config = ExperimentConfig.dict_()
data_config = ExperimentConfig.Data.dict_()
```

### Error Handling Pattern

```python
from hagike.utils import check_path_readable, ensure_path_writable
from hagike.utils import add_msg, MsgLevel, error_proc

try:
    # Check input file
    if check_path_readable("input.txt", is_raise=False):
        with open("input.txt") as f:
            data = f.read()
    
    # Ensure output directory
    ensure_path_writable("results/output.txt")
    
    # Process data...
    add_msg(MsgLevel.Run.value, "Processing complete")
    
except Exception as e:
    add_msg(MsgLevel.Error.value, f"Error: {str(e)}")
    error_proc(is_exit=False)  # Handle gracefully
```

### Image Processing Pipeline

```python
from hagike.basics.cv import *

def process_image(input_path, output_path):
    """Complete image processing pipeline"""
    
    # Load image
    image = ImStd.load(input_path)
    add_msg(MsgLevel.Run.value, f"Loaded image: {input_path}")
    
    # Enhancement pipeline
    enhanced = im_log(image)  # Enhance dark regions
    smoothed = apply_smoothing_filter(enhanced, ImSmooth.gaussian.value)
    
    # Optional: Add noise for robustness testing
    # noisy = add_noise_to_image(smoothed, ImNoise.gaussian.value)
    
    # Histogram equalization
    equalized, orig_hist, eq_hist = flatten_histogram(smoothed)
    
    # Save results
    equalized.save(output_path)
    draw_histogram(eq_hist, save=f"{output_path}_hist.png")
    
    add_msg(MsgLevel.Run.value, f"Saved processed image: {output_path}")

# Usage
process_image("input.jpg", "output.jpg")
```

### Model Training Pattern

```python
from hagike.models.train import TrainerTemp, TrainerInfo
from hagike.models.temp import ModuleNode
import torch.nn as nn

# Define model architecture
class CNNModel(ModuleNode):
    def __init__(self, num_classes=10):
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        super().__init__(model)

# Training configuration
training_config = TrainerInfo.dict_({
    TrainerInfo.device: "cuda",
    TrainerInfo.max_epochs: 50,
    TrainerInfo.learning_rate: 0.001
})

# Create and train model
model = CNNModel(num_classes=10)

# Assuming trainer components are configured
trainer = TrainerTemp(
    model=model,
    optim=optimizer,
    criterion=criterion,
    dataloader=dataloader,
    monitor=monitor,
    evaluator=evaluator,
    info=training_config
)

trainer.train()
```

## Best Practices

### 1. Use Enums for Configuration
```python
# Good: Type-safe, hierarchical configuration
@advanced_enum()
class Config(SuperEnum):
    learning_rate = 0.001
    class Model(SuperEnum):
        layers = [64, 128, 256]

# Avoid: Plain dictionaries
config = {"learning_rate": 0.001, "model_layers": [64, 128, 256]}
```

### 2. Proper Error Handling
```python
# Good: Use framework's error handling
from hagike.utils import add_msg, MsgLevel, error_proc

try:
    # risky operation
    pass
except Exception as e:
    add_msg(MsgLevel.Error.value, f"Operation failed: {e}")
    error_proc(is_exit=False)
```

### 3. Structured Logging
```python
# Good: Structured logging with context
log_entry = LogTemp(
    level=1,
    src="training",
    msg="Epoch completed",
    event_main=epoch_id,
    event_sub=batch_id
)
logger_g.add_log(log_entry)
```

### 4. File Operations Safety
```python
# Good: Always check paths
from hagike.utils import ensure_path_writable, check_path_readable

if check_path_readable(input_path, is_raise=False):
    ensure_path_writable(output_path)
    # proceed with file operations
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure to install all dependencies
   pip install -r requirements.txt
   
   # For development installation
   pip install -e .
   ```

2. **Enum UUID Errors**
   ```python
   # Always use .value when accessing enum members
   add_msg(MsgLevel.Run.value, "message")  # Correct
   add_msg(MsgLevel.Run, "message")        # Wrong
   ```

3. **Device Errors**
   ```python
   # Check CUDA availability
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model.to(device=device)
   ```

4. **Path Issues**
   ```python
   # Use framework's path utilities
   from hagike.utils import ensure_path_writable
   
   ensure_path_writable("results/experiment/model.pth")
   # This creates directories automatically
   ```

## Next Steps

1. **Explore the Full API**: Check `API_DOCUMENTATION.md` for complete reference
2. **Study Examples**: Look at the `demos/` directory for complete examples
3. **Read Source Code**: The framework is well-documented with docstrings
4. **Extend the Framework**: Add your own modules following the established patterns

## Getting Help

- **Documentation**: Read the comprehensive API documentation
- **Source Code**: All modules have detailed docstrings
- **Examples**: Check the `demos/` and `tests/` directories
- **Issues**: Report issues on the GitHub repository

---

Happy coding with Hagike! 🚀
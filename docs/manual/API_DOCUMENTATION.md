# Hagike Toolkit - Comprehensive API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Modules](#core-modules)
4. [Utilities](#utilities)
5. [Computer Vision](#computer-vision)
6. [Machine Learning Models](#machine-learning-models)
7. [Training Framework](#training-framework)
8. [System Components](#system-components)
9. [Tools](#tools)
10. [Logging System](#logging-system)
11. [Examples](#examples)
12. [API Reference](#api-reference)

## Overview

Hagike is a comprehensive toolkit for machine learning development and deployment, developed by hagikehappy at Nanjing University. It provides a unified framework for computer vision, natural language processing, reinforcement learning, and general machine learning tasks.

### Key Features

- **Modular Architecture**: Organized into specialized packages for different domains
- **Advanced Enum System**: Type-safe configuration management
- **Model Templates**: Reusable components for building complex models
- **Training Framework**: Complete training pipeline with monitoring and evaluation
- **Computer Vision Tools**: Image processing and transformation utilities
- **System Integration**: Parallel computing and task management
- **Logging System**: Comprehensive logging and monitoring capabilities

### Framework Philosophy

The framework follows these design principles:

1. **Code Accumulation**: Enable reusable code across different projects
2. **Standardization**: Provide consistent interfaces and patterns
3. **Modularity**: Separate concerns into specialized packages
4. **Extensibility**: Allow easy extension and customization
5. **Integration**: Seamlessly integrate with existing packages

## Installation

### Requirements

- Python >= 3.10
- PyTorch (for deep learning models)
- PIL (for image processing)
- colorama (for colored output)
- loguru (for logging)
- Other dependencies listed in `requirements.txt`

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/hagikehappy/hagike.git
cd hagike

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import hagike
from hagike.utils import *
from hagike.log import logger_g
from hagike.models.temp import ModuleNode
```

## Core Modules

### Package Structure

```
hagike/
├── utils/          # Utility functions and classes
├── log/           # Logging system
├── basics/        # Basic algorithms
│   └── cv/        # Computer vision algorithms
├── models/        # Model framework
│   ├── temp/      # Model templates
│   ├── train/     # Training components
│   ├── cv/        # Computer vision models
│   ├── nlp/       # NLP models
│   └── rl/        # Reinforcement learning models
├── system/        # System components
│   ├── core/      # System core
│   └── interface/ # System interfaces
└── tools/         # Additional tools
    └── matlab/    # MATLAB integration
```

## Utilities

### Message System

The message system provides colored console output with different severity levels.

#### Classes

##### `MsgLevel`
```python
@advanced_enum()
class MsgLevel(SuperEnum):
    Run = Fore.GREEN + "RUN: "
    Warning = Fore.YELLOW + "WARNING: "
    Error = Fore.RED + "ERROR: "
    Panic = Fore.RED + "PANIC: "
```

**Description**: Enumeration of message levels with color-coded prefixes.

**Members**:
- `Run`: Green-colored run messages
- `Warning`: Yellow-colored warnings
- `Error`: Red-colored errors
- `Panic`: Red-colored panic messages

#### Functions

##### `add_msg(level: int, script: str, is_print=True)`

**Description**: Add a colored message to the console.

**Parameters**:
- `level` (int): Message level from MsgLevel enum
- `script` (str): Message text to display
- `is_print` (bool, optional): Whether to print the message. Defaults to True.

**Example**:
```python
from hagike.utils import add_msg, MsgLevel

add_msg(MsgLevel.Run.value, "Processing started")
add_msg(MsgLevel.Warning.value, "Low memory warning")
add_msg(MsgLevel.Error.value, "File not found")
```

##### `error_proc(is_exit=True)`

**Description**: Handle error processing and optionally exit the program.

**Parameters**:
- `is_exit` (bool, optional): Whether to exit the program. Defaults to True.

### Configuration Management

#### Functions

##### `read_json(json_path: str) -> dict`

**Description**: Read and parse a JSON configuration file.

**Parameters**:
- `json_path` (str): Path to the JSON file

**Returns**:
- `dict`: Parsed JSON data

**Raises**:
- `FileNotFoundError`: If the file doesn't exist
- `json.JSONDecodeError`: If the file contains invalid JSON

**Example**:
```python
from hagike.utils import read_json

config = read_json("config.json")
print(config["model_params"])
```

##### `write_json(json_path: str, data: dict, settings: dict | None = None) -> None`

**Description**: Write data to a JSON file with automatic directory creation.

**Parameters**:
- `json_path` (str): Path where to save the JSON file
- `data` (dict): Data to serialize to JSON
- `settings` (dict, optional): Additional JSON serialization settings

**Example**:
```python
from hagike.utils import write_json

config = {"learning_rate": 0.001, "batch_size": 32}
write_json("experiments/config.json", config)
```

### File Operations

#### Exception Classes

##### `FileWritableError`
```python
class FileWritableError(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code
```

**Description**: Exception raised when a file cannot be written to.

##### `FileReadableError`
```python
class FileReadableError(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code
```

**Description**: Exception raised when a file cannot be read.

#### Functions

##### `check_path_readable(path: str, is_raise: bool = True) -> bool`

**Description**: Verify that a path is readable, checking path validity, existence, file type, and permissions.

**Parameters**:
- `path` (str): Path to check
- `is_raise` (bool, optional): Whether to raise exceptions on failure. Defaults to True.

**Returns**:
- `bool`: True if path is readable

**Raises**:
- `FileReadableError`: If path is not readable and `is_raise=True`

**Example**:
```python
from hagike.utils import check_path_readable

if check_path_readable("data/input.txt", is_raise=False):
    with open("data/input.txt", 'r') as f:
        content = f.read()
```

##### `ensure_path_writable(path: str, is_raise: bool = True) -> bool`

**Description**: Ensure a path is writable, creating parent directories if necessary.

**Parameters**:
- `path` (str): Path to check/create
- `is_raise` (bool, optional): Whether to raise exceptions on failure. Defaults to True.

**Returns**:
- `bool`: True if path is writable

**Example**:
```python
from hagike.utils import ensure_path_writable

output_path = "results/experiment_1/model.pth"
if ensure_path_writable(output_path):
    # Save model to path
    pass
```

### Caching System

#### Functions

##### `save_data_to_pkl(data: Any, path: str) -> None`

**Description**: Save data to a pickle file with automatic directory creation.

**Parameters**:
- `data` (Any): Data to serialize
- `path` (str): Path to save the pickle file

**Example**:
```python
from hagike.utils import save_data_to_pkl

results = {"accuracy": 0.95, "loss": 0.05}
save_data_to_pkl(results, "cache/results.pkl")
```

##### `load_data_from_pkl(path: str) -> Any`

**Description**: Load data from a pickle file.

**Parameters**:
- `path` (str): Path to the pickle file

**Returns**:
- `Any`: Deserialized data

**Example**:
```python
from hagike.utils import load_data_from_pkl

results = load_data_from_pkl("cache/results.pkl")
print(f"Accuracy: {results['accuracy']}")
```

#### Classes

##### `CacheTemp`
```python
class CacheTemp:
    def __init__(self, func: Callable, max_size: int, typed: bool = False):
        # Implementation details
    
    def __call__(self, *args, **kwargs):
        # Caching logic
```

**Description**: Template for function result caching.

**Parameters**:
- `func` (Callable): Function to cache
- `max_size` (int): Maximum cache size
- `typed` (bool, optional): Whether to consider argument types. Defaults to False.

### Advanced Enum System

The hagike framework includes a sophisticated enum system that extends Python's basic enums with additional functionality.

#### Core Classes

##### `SuperEnum`

**Description**: Base class for advanced enumeration with UUID-based identification, hierarchical organization, and rich functionality.

**Key Features**:
- UUID-based unique identification
- Hierarchical enum organization
- Index-based ordering
- Value mapping and retrieval
- Tree-like printing and visualization

**Class Methods**:

###### `get_uuid_() -> uuid_t`
**Description**: Get the UUID of the enum class itself.

###### `get_cls_(uuid: uuid_t) -> SuperEnum`
**Description**: Get the subclass corresponding to a UUID.

###### `get_base_(uuid: uuid_t) -> SuperEnum`
**Description**: Get the parent class of an enum member.

###### `get_name_(uuid: uuid_t) -> str`
**Description**: Get the name of an enum member.

###### `get_value_(src: index_t | uuid_t, index_or_uuid: bool = False) -> Any`
**Description**: Get the value of an enum member (returns deep copy).

###### `check_in_(uuid: uuid_t, all_or_index: bool = False, is_raise: bool = True) -> bool`
**Description**: Check if a UUID is contained in the enum.

###### `dict_(enum_dict: Mapping[uuid_t, Any] = None, is_force: bool = True) -> Dict[uuid_t, Any]`
**Description**: Create a complete dictionary with default values for missing enum members.

###### `list_(enum_dict: Mapping[uuid_t, Any] = None, is_default: bool = False) -> List[Any]`
**Description**: Convert enum dictionary to ordered list based on index.

###### `print_() and tree_()`
**Description**: Print enum structure in flat or hierarchical tree format.

**Example Usage**:
```python
from hagike.utils import SuperEnum, advanced_enum

@advanced_enum()
class ModelConfig(SuperEnum):
    learning_rate = 0.001
    batch_size = 32
    epochs = 100
    
    class Optimizer(SuperEnum):
        adam = "adam"
        sgd = "sgd"
        
        class Adam(SuperEnum):
            beta1 = 0.9
            beta2 = 0.999

# Usage
config = ModelConfig.dict_()
lr = ModelConfig.get_value_(ModelConfig.learning_rate)
ModelConfig.tree_(is_value=True)  # Print hierarchical structure
```

##### `@advanced_enum()` Decorator

**Description**: Decorator that automatically configures enum classes with UUID mapping, hierarchy, and advanced features.

**Features**:
- Automatic UUID assignment
- Hierarchy building
- Index management
- Validation and error checking

## Computer Vision

### Image Processing

The computer vision module provides comprehensive image processing capabilities.

#### Core Classes

##### `ImStd`

**Description**: Standard image container class for consistent image handling across the framework.

**Key Features**:
- Multiple format support (PIL, NumPy, etc.)
- Automatic format conversion
- Metadata management
- Processing pipeline integration

#### Image Transformations

##### `transform_curve(im: ImStd, transform: Callable) -> ImStd`

**Description**: Apply intensity transformation to image using a callable function.

**Parameters**:
- `im` (ImStd): Input image
- `transform` (Callable): Transformation function

**Returns**:
- `ImStd`: Transformed image

**Example**:
```python
from hagike.basics.cv import transform_curve, ImStd
import math

def gamma_correction(x):
    return int(255 * ((x / 255.0) ** (1/2.2)))

# Apply gamma correction
corrected_image = transform_curve(image, gamma_correction)
```

##### `im_log(im: ImStd) -> ImStd`

**Description**: Apply logarithmic transformation to enhance dark regions.

**Parameters**:
- `im` (ImStd): Input image (uint8 format)

**Returns**:
- `ImStd`: Log-transformed image

**Example**:
```python
from hagike.basics.cv import im_log

enhanced_image = im_log(dark_image)
```

#### Image Filtering

##### `apply_smoothing_filter(im: ImStd, smooth: uuid_t, para: Mapping[uuid_t, Any] | None = None) -> ImStd`

**Description**: Apply smoothing filters to reduce noise.

**Parameters**:
- `im` (ImStd): Input image
- `smooth` (uuid_t): Filter type from ImSmooth enum
- `para` (Mapping, optional): Filter parameters

**Returns**:
- `ImStd`: Filtered image

**Filter Types**:
- `ImSmooth.mean`: Mean filter
- `ImSmooth.gaussian`: Gaussian filter

**Example**:
```python
from hagike.basics.cv import apply_smoothing_filter, ImSmooth

# Apply Gaussian smoothing
smoothed = apply_smoothing_filter(
    image, 
    ImSmooth.gaussian.value,
    {ImSmooth.gaussian.sigma: 1.5}
)
```

##### `apply_sharpening_filter(im: ImStd, sharp: uuid_t, para: Mapping[uuid_t, Any] | None = None) -> ImStd`

**Description**: Apply sharpening filters to enhance edges.

**Parameters**:
- `im` (ImStd): Input image
- `sharp` (uuid_t): Filter type from ImSharp enum
- `para` (Mapping, optional): Filter parameters

**Filter Types**:
- `ImSharp.laplacian`: Laplacian filter
- `ImSharp.sharpening`: Unsharp masking

#### Histogram Processing

##### `convert_histogram(im: ImStd) -> np.ndarray`

**Description**: Compute image histogram.

**Parameters**:
- `im` (ImStd): Input image

**Returns**:
- `np.ndarray`: Histogram array

##### `flatten_histogram(im: ImStd) -> tuple`

**Description**: Perform histogram equalization.

**Parameters**:
- `im` (ImStd): Input image

**Returns**:
- `tuple`: (equalized_image, original_histogram, equalized_histogram)

**Example**:
```python
from hagike.basics.cv import flatten_histogram

equalized_img, orig_hist, eq_hist = flatten_histogram(image)
```

##### `draw_histogram(histogram, save: None | str = None, show: bool = True) -> None`

**Description**: Visualize histogram with matplotlib.

**Parameters**:
- `histogram`: Histogram data
- `save` (str, optional): Path to save the plot
- `show` (bool): Whether to display the plot

#### Noise Addition

##### `add_noise_to_image(im: ImStd, noise: uuid_t, para: Mapping[uuid_t, Any] | None = None) -> ImStd`

**Description**: Add various types of noise to images for testing and augmentation.

**Parameters**:
- `im` (ImStd): Input image
- `noise` (uuid_t): Noise type from ImNoise enum
- `para` (Mapping, optional): Noise parameters

**Noise Types**:
- `ImNoise.gaussian`: Gaussian noise
- `ImNoise.salt_pepper`: Salt and pepper noise
- `ImNoise.poisson`: Poisson noise

**Example**:
```python
from hagike.basics.cv import add_noise_to_image, ImNoise

# Add Gaussian noise
noisy_image = add_noise_to_image(
    image,
    ImNoise.gaussian.value,
    {ImNoise.gaussian.mean: 0, ImNoise.gaussian.std: 25}
)
```

## Machine Learning Models

### Model Templates

The model framework provides reusable templates for building complex neural networks.

#### Core Classes

##### `ModuleNode`

**Description**: Base class for all neural network modules in the framework.

**Features**:
- PyTorch nn.Module integration
- Automatic device management
- Weight loading/saving
- Mode switching (train/eval)
- Configuration management

**Key Methods**:

###### `__init__(model: nn.Module, info: Dict[uuid_t, Any] | None = None)`
**Description**: Initialize module with PyTorch model and configuration.

###### `load_weights(weights_src: str | Any, is_path: bool = False)`
**Description**: Load model weights from file or memory.

###### `save_weights(path: str | None = None) -> Any`
**Description**: Save model weights to file or return state dict.

###### `to(device: str | None = None, mode: uuid_t | None = None)`
**Description**: Move model to device and/or change mode.

**Example**:
```python
from hagike.models.temp import ModuleNode, ModuleMode
import torch.nn as nn

class MyModel(ModuleNode):
    def __init__(self):
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        super().__init__(model)
    
    def forward(self, x):
        return self._model(x)

# Usage
model = MyModel()
model.to(device="cuda", mode=ModuleMode.train.value)
model.save_weights("model.pth")
```

##### `ModelTemp`

**Description**: Template for building complex models using directed acyclic graphs (DAG) of modules.

**Features**:
- DAG-based model construction
- Node dependency management
- Automatic execution ordering
- Modular component organization

**Constructor Parameters**:
- `nodes` (Mapping[node_t, ModuleNode]): Mapping of node names to modules
- `deps` (Dict[node_t, Tuple[List[node_t], List[node_t]]]): Dependency relationships
- `info` (Dict[uuid_t, Any], optional): Model configuration

**Example**:
```python
from hagike.models.temp import ModelTemp, ModuleNode
import torch.nn as nn

# Define individual modules
encoder = ModuleNode(nn.Linear(784, 256))
decoder = ModuleNode(nn.Linear(256, 784))

# Define model structure
nodes = {
    "encoder": encoder,
    "decoder": decoder
}

deps = {
    "_in": ([], ["encoder"]),        # Input connects to encoder
    "encoder": (["_in"], ["decoder"]), # Encoder to decoder
    "decoder": (["encoder"], ["_out"]), # Decoder to output
    "_out": (["decoder"], [])        # Output node
}

# Create model
autoencoder = ModelTemp(nodes, deps)

# Use model
output = autoencoder(input_tensor)
```

### Training Framework

The training framework provides a complete pipeline for model training with monitoring and evaluation.

#### Core Classes

##### `TrainerTemp`

**Description**: Main training orchestrator that coordinates all training components.

**Components**:
- Model management
- Optimizer integration
- Loss computation
- Data loading
- Training monitoring
- Model evaluation

**Constructor Parameters**:
- `model` (ModuleNode): Model to train
- `optim` (OptimTemp): Optimizer configuration
- `criterion` (CriterionTemp): Loss function
- `dataloader` (DataLoaderTemp): Data management
- `monitor` (TrainerMonitor): Training monitoring
- `evaluator` (EvaluatorTemp): Model evaluation
- `info` (Dict[uuid_t, Any], optional): Training configuration

**Key Methods**:

###### `train()`
**Description**: Execute the complete training loop with monitoring and checkpointing.

**Training Process**:
1. Initialize monitoring
2. For each epoch:
   - Set model to training mode
   - Process training batches
   - Compute and backpropagate loss
   - Update model parameters
   - Evaluate on validation set
   - Record metrics and save best model
3. Finalize training

**Example**:
```python
from hagike.models.train import TrainerTemp, TrainerInfo
from hagike.models.temp import ModuleNode
import torch.nn as nn

# Define model
model = ModuleNode(nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
))

# Configure training
config = {
    TrainerInfo.device: "cuda",
    TrainerInfo.max_epochs: 100,
    TrainerInfo.learning_rate: 0.001
}

# Create trainer (assuming other components are configured)
trainer = TrainerTemp(
    model=model,
    optim=optimizer,
    criterion=loss_fn,
    dataloader=data_loader,
    monitor=monitor,
    evaluator=evaluator,
    info=config
)

# Start training
trainer.train()
```

##### Configuration Enums

The training framework uses several enum classes for configuration:

###### `TrainerInfo`
```python
@advanced_enum()
class TrainerInfo(SuperEnum):
    device = "cuda"
    max_epochs = 100
    learning_rate = 0.001
    # ... other training parameters
```

###### `EvaluatorInfo`
```python
@advanced_enum()
class EvaluatorInfo(SuperEnum):
    metrics = ["accuracy", "f1_score"]
    validation_frequency = 1
    # ... other evaluation parameters
```

## System Components

### Parallel Computing System

The system module provides task-level parallel computing capabilities.

#### Architecture

```
system/
├── core/          # System kernel
└── interface/     # System interfaces
```

**Features**:
- Task-level parallelization
- Resource management
- Interface templates
- Scalable computation

## Tools

### MATLAB Integration

The tools module includes MATLAB engine integration for scientific computing.

#### Classes

##### `MEngine`

**Description**: MATLAB engine wrapper for seamless Python-MATLAB integration.

**Features**:
- MATLAB function calling
- Variable exchange
- Object manipulation
- Error handling

**Key Methods**:

###### `call(script: str, *args, num: int = -1) -> Any`
**Description**: Execute MATLAB function with arguments.

###### `obj_call(obj, script: str, *args, num: int = -1) -> Any`
**Description**: Call method on MATLAB object.

**Example**:
```python
from hagike.tools.matlab import MEngine

engine = MEngine()
result = engine.call("sin", 3.14159)
matrix = engine.call("rand", 3, 3)
```

##### `MSys`

**Description**: MATLAB system analysis tools.

**Features**:
- System response analysis
- Transfer function manipulation
- Control system design
- Performance metrics

## Logging System

### Core Components

The logging system provides comprehensive logging capabilities with global configuration.

#### Classes

##### `LoggerTemp`

**Description**: Advanced logging manager with configurable output and recording.

**Features**:
- Global configuration
- Structured log format
- Multiple output targets
- Log analysis capabilities
- Cross-process access

**Key Methods**:

###### `__init__(conf: Dict[uuid_t, Any] | None = None, is_init: bool = True)`
**Description**: Initialize logger with configuration.

###### `add_log(log: LogTemp, is_print: bool = True)`
**Description**: Add structured log entry.

###### `print_msg(msg: str)`
**Description**: Print message without logging.

**Example**:
```python
from hagike.log import logger_g, LogTemp

# Use global logger
log_entry = LogTemp(
    level=1,
    src="training",
    msg="Training started",
    event_main=100,
    event_sub=1
)

logger_g.add_log(log_entry)
```

##### `LogTemp`

**Description**: Structured log entry data class.

**Fields**:
- `uuid` (int): Log identifier
- `time` (int | float): Timestamp
- `level` (uuid_t): Log level
- `src` (str): Source component
- `event_main` (uuid_t): Main event ID
- `event_sub` (uuid_t): Sub-event ID
- `msg` (str): Log message
- `else__` (Any): Additional data

#### Configuration

##### `LoggerConf`
```python
@advanced_enum()
class LoggerConf(SuperEnum):
    # Configuration options for logging behavior
    output_console = True
    output_file = True
    log_level = 1
    # ... other configuration options
```

## Examples

### Basic Usage Example

```python
import hagike
from hagike.utils import add_msg, MsgLevel, read_json
from hagike.log import logger_g
from hagike.models.temp import ModuleNode, ModuleMode

# Message system
add_msg(MsgLevel.Run.value, "Application started")

# Configuration management
config = read_json("config.json")

# Logging
logger_g.print_msg("Loading model...")

# Model creation
import torch.nn as nn
model = ModuleNode(nn.Linear(784, 10))
model.to(device="cuda", mode=ModuleMode.train.value)

add_msg(MsgLevel.Run.value, "Setup complete")
```

### Computer Vision Pipeline

```python
from hagike.basics.cv import *

# Load and process image
image = ImStd.load("input.jpg")

# Apply transformations
enhanced = im_log(image)
smoothed = apply_smoothing_filter(enhanced, ImSmooth.gaussian.value)

# Add noise for testing
noisy = add_noise_to_image(smoothed, ImNoise.gaussian.value)

# Analyze histogram
hist = convert_histogram(noisy)
draw_histogram(hist, save="histogram.png")

# Equalize histogram
equalized, _, _ = flatten_histogram(noisy)
```

### Training Pipeline Example

```python
from hagike.models.train import *
from hagike.models.temp import ModuleNode
import torch.nn as nn

# Define model
model = ModuleNode(nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10)
))

# Configure training components
trainer_config = TrainerInfo.dict_({
    TrainerInfo.device: "cuda",
    TrainerInfo.max_epochs: 50,
    TrainerInfo.learning_rate: 0.001
})

# Create and run trainer
trainer = TrainerTemp(
    model=model,
    optim=optimizer,
    criterion=criterion,
    dataloader=dataloader,
    monitor=monitor,
    evaluator=evaluator,
    info=trainer_config
)

trainer.train()
```

### Advanced Enum Usage

```python
from hagike.utils import SuperEnum, advanced_enum

@advanced_enum()
class ExperimentConfig(SuperEnum):
    _sequence = ("model", "data", "training", "evaluation")
    
    class Model(SuperEnum):
        architecture = "resnet50"
        pretrained = True
        
        class ResNet(SuperEnum):
            layers = [3, 4, 6, 3]
            num_classes = 1000
    
    class Data(SuperEnum):
        batch_size = 32
        num_workers = 4
        augmentation = True
    
    class Training(SuperEnum):
        epochs = 100
        learning_rate = 0.001
        optimizer = "adam"
    
    class Evaluation(SuperEnum):
        metrics = ["accuracy", "f1_score"]
        save_best = True

# Usage
config = ExperimentConfig.dict_()
model_config = ExperimentConfig.Model.dict_()
ExperimentConfig.tree_(is_value=True)
```

## API Reference

### Complete Function Index

#### Utils Module
- `add_msg(level, script, is_print=True)` - Add colored console message
- `error_proc(is_exit=True)` - Handle error processing
- `read_json(json_path)` - Read JSON configuration file
- `write_json(json_path, data, settings=None)` - Write JSON configuration
- `check_path_readable(path, is_raise=True)` - Verify path readability
- `ensure_path_writable(path, is_raise=True)` - Ensure path writability
- `save_data_to_pkl(data, path)` - Save data to pickle file
- `load_data_from_pkl(path)` - Load data from pickle file

#### Computer Vision Module
- `transform_curve(im, transform)` - Apply intensity transformation
- `im_log(im)` - Logarithmic transformation
- `apply_smoothing_filter(im, smooth, para=None)` - Apply smoothing filters
- `apply_sharpening_filter(im, sharp, para=None)` - Apply sharpening filters
- `frequency_domain_filtering(im, freq, para=None)` - Frequency domain filtering
- `convert_histogram(im)` - Compute image histogram
- `flatten_histogram(im)` - Histogram equalization
- `draw_histogram(histogram, save=None, show=True)` - Visualize histogram
- `add_noise_to_image(im, noise, para=None)` - Add noise to image

#### Model Framework
- `ModuleNode.__init__(model, info=None)` - Initialize module node
- `ModuleNode.load_weights(weights_src, is_path=False)` - Load model weights
- `ModuleNode.save_weights(path=None)` - Save model weights
- `ModuleNode.to(device=None, mode=None)` - Move model and set mode
- `ModelTemp.__init__(nodes, deps, info=None)` - Initialize DAG model
- `ModelTemp.forward(*args)` - Execute model forward pass

#### Training Framework
- `TrainerTemp.__init__(model, optim, criterion, dataloader, monitor, evaluator, info=None)` - Initialize trainer
- `TrainerTemp.train()` - Execute training loop
- `TrainerTemp.end()` - Cleanup training resources

#### Logging System
- `LoggerTemp.__init__(conf=None, is_init=True)` - Initialize logger
- `LoggerTemp.add_log(log, is_print=True)` - Add log entry
- `LoggerTemp.print_msg(msg)` - Print message without logging
- `LoggerTemp.init()` - Start logger
- `LoggerTemp.end()` - Stop logger

#### Enum System
- `SuperEnum.get_uuid_()` - Get enum class UUID
- `SuperEnum.get_cls_(uuid)` - Get subclass by UUID
- `SuperEnum.get_name_(uuid)` - Get enum member name
- `SuperEnum.get_value_(src, index_or_uuid=False)` - Get enum member value
- `SuperEnum.check_in_(uuid, all_or_index=False, is_raise=True)` - Check UUID membership
- `SuperEnum.dict_(enum_dict=None, is_force=True)` - Create complete enum dictionary
- `SuperEnum.list_(enum_dict=None, is_default=False)` - Convert to ordered list
- `SuperEnum.print_(is_value=False)` - Print enum structure
- `SuperEnum.tree_(is_value=False)` - Print hierarchical structure

### Class Hierarchy

```
SuperEnum
├── MsgLevel
├── LoggerConf
├── LoggerStatus
├── DataSrc
├── DataForm
├── ImStyle
├── ImScale
├── ImColor
├── ImNoise
├── ImSmooth
├── ImSharp
├── TrainerInfo
├── EvaluatorInfo
└── ...

nn.Module
└── ModuleNode
    ├── ModelTemp
    ├── ModuleTemp
    └── Custom Models

Exception
├── FileWritableError
├── FileReadableError
├── EnumOccupiedError
├── ModelError
├── TrainingError
└── ...
```

---

## Contributing

To contribute to the hagike framework:

1. Follow the established coding patterns and documentation standards
2. Use the advanced enum system for configuration management
3. Implement proper error handling with custom exception classes
4. Add comprehensive docstrings with examples
5. Include unit tests for new functionality
6. Update this documentation for new APIs

## License

This project is licensed under the GNU Lesser General Public License v3 (LGPLv3).

---

**Author**: hagikehappy  
**Institution**: School of Electronic Science and Engineering, Nanjing University  
**Repository**: https://github.com/hagikehappy/hagike
# Hagike Toolkit - Function & Class Reference

## Table of Contents

- [Utils Module](#utils-module)
- [Logging System](#logging-system)
- [Computer Vision](#computer-vision)
- [Model Framework](#model-framework)
- [Training Framework](#training-framework)
- [Tools](#tools)
- [Data Handling](#data-handling)

---

## Utils Module

### Message System (`hagike.utils.message`)

#### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `add_msg` | `(level: int, script: str, is_print=True)` | Add colored console message |
| `error_proc` | `(is_exit=True)` | Handle error processing |

#### Classes

| Class | Description | Key Methods |
|-------|-------------|-------------|
| `MsgLevel` | Message level enumeration | `Run`, `Warning`, `Error`, `Panic` |

#### Example Usage
```python
from hagike.utils import add_msg, MsgLevel

add_msg(MsgLevel.Run.value, "Processing started")
add_msg(MsgLevel.Error.value, "File not found")
```

---

### Configuration Management (`hagike.utils.config`)

#### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `read_json` | `(json_path: str) -> dict` | Read JSON configuration file |
| `write_json` | `(json_path: str, data: dict, settings: dict = None)` | Write JSON file |

#### Example Usage
```python
from hagike.utils import read_json, write_json

config = read_json("config.json")
write_json("output.json", {"key": "value"})
```

---

### File Operations (`hagike.utils.file`)

#### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `check_path_readable` | `(path: str, is_raise: bool = True) -> bool` | Verify path readability |
| `ensure_path_writable` | `(path: str, is_raise: bool = True) -> bool` | Ensure path writability |

#### Exception Classes

| Exception | Description |
|-----------|-------------|
| `FileWritableError` | Raised when file cannot be written |
| `FileReadableError` | Raised when file cannot be read |

#### Example Usage
```python
from hagike.utils import check_path_readable, ensure_path_writable

if check_path_readable("input.txt"):
    ensure_path_writable("output.txt")
    # Process files
```

---

### Caching System (`hagike.utils.cache`)

#### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `save_data_to_pkl` | `(data: Any, path: str)` | Save data to pickle file |
| `load_data_from_pkl` | `(path: str) -> Any` | Load data from pickle |

#### Classes

| Class | Constructor | Description |
|-------|-------------|-------------|
| `CacheTemp` | `(func: Callable, max_size: int, typed: bool = False)` | Function result caching |

#### Example Usage
```python
from hagike.utils import save_data_to_pkl, load_data_from_pkl

save_data_to_pkl({"results": [1, 2, 3]}, "cache.pkl")
data = load_data_from_pkl("cache.pkl")
```

---

### Advanced Enum System (`hagike.utils.enum`)

#### Core Classes

| Class | Description | Key Features |
|-------|-------------|--------------|
| `SuperEnum` | Advanced enumeration base class | UUID-based, hierarchical, rich functionality |

#### SuperEnum Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_uuid_` | `() -> uuid_t` | Get enum class UUID |
| `get_cls_` | `(uuid: uuid_t) -> SuperEnum` | Get subclass by UUID |
| `get_base_` | `(uuid: uuid_t) -> SuperEnum` | Get parent class |
| `get_name_` | `(uuid: uuid_t) -> str` | Get member name |
| `get_value_` | `(src: uuid_t, index_or_uuid: bool = False) -> Any` | Get member value |
| `check_in_` | `(uuid: uuid_t, all_or_index: bool = False, is_raise: bool = True) -> bool` | Check UUID membership |
| `dict_` | `(enum_dict: Mapping = None, is_force: bool = True) -> Dict` | Create complete dictionary |
| `list_` | `(enum_dict: Mapping = None, is_default: bool = False) -> List` | Convert to ordered list |
| `print_` | `(is_value: bool = False)` | Print enum structure |
| `tree_` | `(is_value: bool = False)` | Print hierarchical structure |

#### Decorators

| Decorator | Description |
|-----------|-------------|
| `@advanced_enum()` | Configure enum with UUID mapping and hierarchy |

#### Exception Classes

| Exception | Description |
|-----------|-------------|
| `EnumOccupiedError` | Keyword occupation error |
| `EnumSequenceError` | Sequence access error |
| `EnumTypeError` | Type configuration error |
| `EnumUuidError` | UUID not found error |

#### Example Usage
```python
from hagike.utils import SuperEnum, advanced_enum

@advanced_enum()
class Config(SuperEnum):
    learning_rate = 0.001
    batch_size = 32
    
    class Model(SuperEnum):
        layers = [64, 128, 256]

# Usage
config = Config.dict_()
lr = Config.get_value_(Config.learning_rate)
Config.tree_(is_value=True)
```

---

## Logging System

### Core Components (`hagike.log`)

#### Classes

| Class | Constructor | Description |
|-------|-------------|-------------|
| `LoggerTemp` | `(conf: Dict = None, is_init: bool = True)` | Advanced logging manager |
| `LogTemp` | Data class | Structured log entry |

#### LoggerTemp Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `init` | `()` | Start logger |
| `end` | `()` | Stop logger |
| `add_log` | `(log: LogTemp, is_print: bool = True)` | Add structured log |
| `print_msg` | `(msg: str)` | Print message without logging |
| `print_log` | `(log: LogTemp)` | Print log entry |

#### LogTemp Fields

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | int | Log identifier |
| `time` | int/float | Timestamp |
| `level` | uuid_t | Log level |
| `src` | str | Source component |
| `event_main` | uuid_t | Main event ID |
| `event_sub` | uuid_t | Sub-event ID |
| `msg` | str | Log message |
| `else__` | Any | Additional data |

#### Configuration Enums

| Enum | Description |
|------|-------------|
| `LoggerConf` | Logger configuration options |
| `LoggerStatus` | Logger status states |

#### Global Logger

| Variable | Type | Description |
|----------|------|-------------|
| `logger_g` | LoggerTemp | Global logger instance |

#### Example Usage
```python
from hagike.log import logger_g, LogTemp

log_entry = LogTemp(
    level=1,
    src="training",
    msg="Epoch completed",
    event_main=100
)

logger_g.add_log(log_entry)
logger_g.print_msg("Simple message")
```

---

## Computer Vision

### Image Processing (`hagike.basics.cv`)

#### Core Classes

| Class | Description |
|-------|-------------|
| `ImStd` | Standard image container |

#### Image Transformations

| Function | Signature | Description |
|----------|-----------|-------------|
| `transform_curve` | `(im: ImStd, transform: Callable) -> ImStd` | Apply intensity transformation |
| `im_log` | `(im: ImStd) -> ImStd` | Logarithmic transformation |

#### Image Filtering

| Function | Signature | Description |
|----------|-----------|-------------|
| `apply_smoothing_filter` | `(im: ImStd, smooth: uuid_t, para: Mapping = None) -> ImStd` | Apply smoothing filters |
| `apply_sharpening_filter` | `(im: ImStd, sharp: uuid_t, para: Mapping = None) -> ImStd` | Apply sharpening filters |
| `frequency_domain_filtering` | `(im: ImStd, freq: uuid_t, para: Mapping = None) -> ImStd` | Frequency domain filtering |

#### Filter Enums

| Enum | Members | Description |
|------|---------|-------------|
| `ImSmooth` | `mean`, `gaussian` | Smoothing filter types |
| `ImSharp` | `laplacian`, `sharpening` | Sharpening filter types |
| `ImFreq` | `LPF`, `HPF` | Frequency domain filters |

#### Histogram Processing

| Function | Signature | Description |
|----------|-----------|-------------|
| `convert_histogram` | `(im: ImStd) -> np.ndarray` | Compute image histogram |
| `flatten_histogram` | `(im: ImStd) -> tuple` | Histogram equalization |
| `draw_histogram` | `(histogram, save: str = None, show: bool = True)` | Visualize histogram |
| `draw_cdf` | `(raw_cdf, fla_cdf, ide_cdf, save: str = None, show: bool = True)` | Draw CDF plots |

#### Noise Addition

| Function | Signature | Description |
|----------|-----------|-------------|
| `add_noise_to_image` | `(im: ImStd, noise: uuid_t, para: Mapping = None) -> ImStd` | Add noise to image |

#### Noise Types

| Enum | Members | Description |
|------|---------|-------------|
| `ImNoise` | `gaussian`, `salt_pepper`, `poisson` | Noise types |

#### Exception Classes

| Exception | Description |
|-----------|-------------|
| `ImageFilterError` | Image filtering errors |
| `ImageNoiseError` | Image noise processing errors |

#### Example Usage
```python
from hagike.basics.cv import *

# Load and process image
image = ImStd.load("input.jpg")
enhanced = im_log(image)
smoothed = apply_smoothing_filter(enhanced, ImSmooth.gaussian.value)
hist = convert_histogram(image)
draw_histogram(hist, save="histogram.png")
```

---

## Model Framework

### Model Templates (`hagike.models.temp`)

#### Core Classes

| Class | Constructor | Description |
|-------|-------------|-------------|
| `ModuleNode` | `(model: nn.Module, info: Dict = None)` | Base neural network module |
| `ModelTemp` | `(nodes: Mapping, deps: Dict, info: Dict = None)` | DAG-based model template |
| `ModuleTemp` | `(model: nn.Module, info: Dict = None)` | Generic module template |

#### ModuleNode Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `load_weights` | `(weights_src: str/Any, is_path: bool = False)` | Load model weights |
| `save_weights` | `(path: str = None) -> Any` | Save model weights |
| `to` | `(device: str = None, mode: uuid_t = None)` | Move model and set mode |
| `forward` | `(*args)` | Forward pass |

#### ModelTemp Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(*args) -> Tuple/Any` | Execute DAG forward pass |
| `module` | `(key: node_t) -> ModuleNode` | Get module by key |
| `print_model` | `(blank: int = 0)` | Print model structure |
| `update` | `()` | Update model structure |

#### Configuration Enums

| Enum | Members | Description |
|------|---------|-------------|
| `ModuleMode` | `train`, `eval` | Module modes |
| `ModuleKey` | Various keys | Module configuration keys |
| `ModuleInfo` | Various info | Module information fields |

#### Exception Classes

| Exception | Description |
|-----------|-------------|
| `ModelError` | Model-related errors |
| `ModelWarning` | Model warnings |

#### Example Usage
```python
from hagike.models.temp import ModuleNode, ModelTemp, ModuleMode
import torch.nn as nn

# Simple module
class MyModel(ModuleNode):
    def __init__(self):
        model = nn.Sequential(nn.Linear(784, 10))
        super().__init__(model)

model = MyModel()
model.to(device="cuda", mode=ModuleMode.train.value)

# DAG model
nodes = {"encoder": encoder_module, "decoder": decoder_module}
deps = {
    "_in": ([], ["encoder"]),
    "encoder": (["_in"], ["decoder"]),
    "decoder": (["encoder"], ["_out"]),
    "_out": (["decoder"], [])
}
dag_model = ModelTemp(nodes, deps)
```

---

## Training Framework

### Training Components (`hagike.models.train`)

#### Core Classes

| Class | Constructor | Description |
|-------|-------------|-------------|
| `TrainerTemp` | `(model, optim, criterion, dataloader, monitor, evaluator, info)` | Main training orchestrator |
| `OptimTemp` | Optimizer template | Optimizer wrapper |
| `CriterionTemp` | Criterion template | Loss function wrapper |
| `DataLoaderTemp` | DataLoader template | Data loading wrapper |
| `TrainerMonitor` | Monitor template | Training monitoring |
| `EvaluatorTemp` | Evaluator template | Model evaluation |

#### TrainerTemp Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `train` | `()` | Execute training loop |
| `end` | `()` | Cleanup resources |

#### Dataset Classes

| Class | Description |
|-------|-------------|
| `DatasetTemp` | Base dataset template |
| `TcvDataset` | Computer vision dataset |

#### Configuration Enums

| Enum | Description |
|------|-------------|
| `TrainerInfo` | Training configuration |
| `EvaluatorInfo` | Evaluation configuration |
| `MonitorInfo` | Monitoring configuration |
| `DatasetInfo` | Dataset configuration |
| `DataLoaderInfo` | DataLoader configuration |
| `OptimInfo` | Optimizer configuration |
| `CriterionInfo` | Criterion configuration |

#### Exception Classes

| Exception | Description |
|-----------|-------------|
| `TrainingError` | Training-related errors |
| `TrainingWarning` | Training warnings |

#### Example Usage
```python
from hagike.models.train import TrainerTemp, TrainerInfo

config = TrainerInfo.dict_({
    TrainerInfo.device: "cuda",
    TrainerInfo.max_epochs: 100,
    TrainerInfo.learning_rate: 0.001
})

trainer = TrainerTemp(
    model=model,
    optim=optimizer,
    criterion=criterion,
    dataloader=dataloader,
    monitor=monitor,
    evaluator=evaluator,
    info=config
)

trainer.train()
```

---

## Tools

### MATLAB Integration (`hagike.tools.matlab`)

#### Engine Classes

| Class | Constructor | Description |
|-------|-------------|-------------|
| `MEngine` | `()` | MATLAB engine wrapper |
| `MSys` | `(m: MEngine, ...)` | MATLAB system analysis |
| `MSysConst` | `(m: Any, amp: float = 1.0)` | Constant system |

#### MEngine Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `call` | `(script: str, *args, num: int = -1) -> Any` | Execute MATLAB function |
| `obj_call` | `(obj, script: str, *args, num: int = -1) -> Any` | Call object method |
| `obj_value` | `(obj: Any, script: str) -> Any` | Get object property |
| `print` | `(*args)` | Print to MATLAB console |
| `operator` | `(script: uuid_t, *args)` | Execute operator |
| `exit` | `()` | Exit MATLAB engine |

#### MSys Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `gen_response` | `(t: List[float], input_t: uuid_t, input_s: List[float] = None) -> List[float]` | Generate system response |
| `plot_response` | `(para: Dict = None, save_path: str = None)` | Plot response |
| `overshoot` | `() -> float` | Calculate overshoot |
| `settlingtime` | `(delta: float = 0.05) -> float` | Calculate settling time |

#### Configuration Enums

| Enum | Members | Description |
|------|---------|-------------|
| `MCall` | Various call types | MATLAB call types |
| `MOperator` | `basic`, `dot`, `matrix`, `logic` | MATLAB operators |
| `MSysInput` | Input types | System input types |

#### Exception Classes

| Exception | Description |
|-----------|-------------|
| `MEngineCallError` | MATLAB engine call errors |
| `MSysInputError` | System input errors |
| `MSysResponseError` | System response errors |

#### Example Usage
```python
from hagike.tools.matlab import MEngine, MSys

engine = MEngine()
result = engine.call("sin", 3.14159)
matrix = engine.call("rand", 3, 3)

# System analysis
sys = MSys(engine, transfer_function_params)
response = sys.gen_response([0, 1, 2, 3], input_type)
overshoot = sys.overshoot()
```

---

## Data Handling

### Data Packaging (`hagike.utils.data`)

#### Core Classes

| Class | Constructor | Description |
|-------|-------------|-------------|
| `DataPack` | `(data: Any, src: uuid_t = DataSrc.mem, form: uuid_t = DataForm.direct, ...)` | Data container with metadata |

#### DataPack Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_mem` | `() -> bool` | Check if data is in memory |
| `is_direct` | `() -> bool` | Check if data is directly accessible |
| `is_usable` | `() -> bool` | Check if data is ready to use |
| `check` | `()` | Validate data state |
| `use` | `()` | Prepare data for use |
| `load` | `()` | Load data from source |
| `transfer` | `()` | Transfer data between formats |
| `to` | `()` | Convert data format |

#### Configuration Enums

| Enum | Members | Description |
|------|---------|-------------|
| `DataSrc` | `mem`, `file`, `url` | Data source types |
| `DataForm` | `direct`, `compressed`, `encoded` | Data format types |
| `DataCompress` | Compression types | Data compression options |
| `DataInfo` | Information fields | Data metadata fields |

#### Exception Classes

| Exception | Description |
|-----------|-------------|
| `DataPackError` | Data packaging errors |
| `DataPackWarning` | Data packaging warnings |

#### Example Usage
```python
from hagike.utils.data import DataPack, DataSrc, DataForm

# Create data pack
data_pack = DataPack(
    data=my_data,
    src=DataSrc.mem.value,
    form=DataForm.direct.value
)

# Check and use data
if data_pack.is_usable():
    processed_data = process(data_pack.data)
```

---

## Summary

This reference covers all major functions and classes in the Hagike toolkit. Each component is designed to work together seamlessly while maintaining modularity and extensibility. For detailed usage examples and advanced patterns, refer to the main API documentation and quick start guide.

### Key Design Patterns

1. **Enum-based Configuration**: Use `@advanced_enum()` for type-safe, hierarchical configuration
2. **Template Classes**: Inherit from framework templates for consistent interfaces
3. **Error Handling**: Use framework-specific exceptions and error handling utilities
4. **Logging**: Implement structured logging with the global logger system
5. **File Operations**: Always use framework utilities for safe file operations

### Import Patterns

```python
# Core utilities
from hagike.utils import *

# Specific modules
from hagike.log import logger_g, LogTemp
from hagike.models.temp import ModuleNode, ModelTemp
from hagike.models.train import TrainerTemp, TrainerInfo
from hagike.basics.cv import *
```

---

For the most up-to-date information, always refer to the source code docstrings and the comprehensive API documentation.
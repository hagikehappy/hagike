# Hagike Toolkit - Documentation Index

Welcome to the comprehensive documentation for the Hagike toolkit! This index will help you find the information you need quickly.

## 📚 Documentation Structure

### 🚀 Getting Started

| Document | Description | Best For |
|----------|-------------|----------|
| **[Quick Start Guide](QUICK_START_GUIDE.md)** | 5-minute introduction with practical examples | New users, quick setup |
| **[Installation Guide](#installation)** | Detailed installation instructions | First-time setup |

### 📖 Complete References

| Document | Description | Best For |
|----------|-------------|----------|
| **[API Documentation](API_DOCUMENTATION.md)** | Comprehensive guide to all APIs, functions, and components | In-depth understanding, development |
| **[Function Reference](FUNCTION_REFERENCE.md)** | Organized lookup table of all functions and classes | Quick reference, API lookup |

### 🔧 Existing Documentation

| Document | Description | Location |
|----------|-------------|----------|
| **HTML Documentation** | Auto-generated API docs | `docs/hagike/index.html` |
| **README** | Project overview and basic info | `README.md` |
| **Documentation Guide** | pdoc usage and formatting | `docs/README.md` |

## 🎯 Choose Your Path

### I'm New to Hagike
1. **Start Here**: [Quick Start Guide](QUICK_START_GUIDE.md)
   - 5-minute setup and basic examples
   - Common patterns and best practices
   - Troubleshooting guide

2. **Next**: [API Documentation](API_DOCUMENTATION.md) - Overview section
   - Framework philosophy and design principles
   - Package structure and organization

### I Need Specific Information
- **Looking up a function?** → [Function Reference](FUNCTION_REFERENCE.md)
- **Want detailed examples?** → [API Documentation](API_DOCUMENTATION.md)
- **Need installation help?** → [Quick Start Guide](QUICK_START_GUIDE.md) - Installation section

### I'm Developing with Hagike
1. **Core Concepts**: [API Documentation](API_DOCUMENTATION.md)
   - Advanced Enum System
   - Model Templates
   - Training Framework

2. **Quick Reference**: [Function Reference](FUNCTION_REFERENCE.md)
   - Organized by module
   - Function signatures and descriptions
   - Import patterns

## 📋 Documentation by Topic

### 🛠️ Utilities & Core Features

| Topic | Quick Start | Full Documentation | Reference |
|-------|-------------|-------------------|-----------|
| **Message System** | [5-min guide](QUICK_START_GUIDE.md#1-basic-utilities) | [Utilities section](API_DOCUMENTATION.md#utilities) | [Utils module](FUNCTION_REFERENCE.md#utils-module) |
| **Configuration** | [Config example](QUICK_START_GUIDE.md#2-configuration-management) | [Config management](API_DOCUMENTATION.md#configuration-management) | [Config functions](FUNCTION_REFERENCE.md#configuration-management-hagikeutilsconfig) |
| **Advanced Enums** | [Enum example](QUICK_START_GUIDE.md#3-advanced-enums) | [Enum system](API_DOCUMENTATION.md#advanced-enum-system) | [Enum reference](FUNCTION_REFERENCE.md#advanced-enum-system-hagikeutilsenum) |
| **File Operations** | [Error handling](QUICK_START_GUIDE.md#error-handling-pattern) | [File operations](API_DOCUMENTATION.md#file-operations) | [File functions](FUNCTION_REFERENCE.md#file-operations-hagikeutilsfile) |
| **Logging** | [Logging example](QUICK_START_GUIDE.md#6-logging-system) | [Logging system](API_DOCUMENTATION.md#logging-system) | [Log reference](FUNCTION_REFERENCE.md#logging-system) |

### 🖼️ Computer Vision

| Topic | Quick Start | Full Documentation | Reference |
|-------|-------------|-------------------|-----------|
| **Image Processing** | [CV pipeline](QUICK_START_GUIDE.md#4-computer-vision) | [Computer vision](API_DOCUMENTATION.md#computer-vision) | [CV reference](FUNCTION_REFERENCE.md#computer-vision) |
| **Image Transformations** | [Transform example](QUICK_START_GUIDE.md#image-processing-pipeline) | [Transformations](API_DOCUMENTATION.md#image-transformations) | [Transform functions](FUNCTION_REFERENCE.md#image-transformations) |
| **Filtering** | [Filter example](QUICK_START_GUIDE.md#4-computer-vision) | [Image filtering](API_DOCUMENTATION.md#image-filtering) | [Filter reference](FUNCTION_REFERENCE.md#image-filtering) |
| **Histograms** | [Histogram example](QUICK_START_GUIDE.md#image-processing-pipeline) | [Histogram processing](API_DOCUMENTATION.md#histogram-processing) | [Histogram functions](FUNCTION_REFERENCE.md#histogram-processing) |

### 🤖 Machine Learning

| Topic | Quick Start | Full Documentation | Reference |
|-------|-------------|-------------------|-----------|
| **Model Framework** | [Model example](QUICK_START_GUIDE.md#5-model-framework) | [Model templates](API_DOCUMENTATION.md#machine-learning-models) | [Model reference](FUNCTION_REFERENCE.md#model-framework) |
| **Training** | [Training example](QUICK_START_GUIDE.md#model-training-pattern) | [Training framework](API_DOCUMENTATION.md#training-framework) | [Training reference](FUNCTION_REFERENCE.md#training-framework) |
| **Model Templates** | [DAG example](QUICK_START_GUIDE.md#5-model-framework) | [Model templates](API_DOCUMENTATION.md#model-templates) | [Template classes](FUNCTION_REFERENCE.md#model-templates-hagikemodelstemp) |

### 🔧 Tools & Integration

| Topic | Quick Start | Full Documentation | Reference |
|-------|-------------|-------------------|-----------|
| **MATLAB Integration** | Not covered | [MATLAB tools](API_DOCUMENTATION.md#tools) | [MATLAB reference](FUNCTION_REFERENCE.md#tools) |
| **System Components** | Not covered | [System components](API_DOCUMENTATION.md#system-components) | Limited coverage |

## 🔍 Search Guide

### By Use Case

**"I want to..."**
- **Get started quickly** → [Quick Start Guide](QUICK_START_GUIDE.md)
- **Learn the framework philosophy** → [API Documentation](API_DOCUMENTATION.md#framework-philosophy)
- **Find a specific function** → [Function Reference](FUNCTION_REFERENCE.md)
- **See practical examples** → [API Documentation](API_DOCUMENTATION.md#examples)
- **Understand the architecture** → [API Documentation](API_DOCUMENTATION.md#core-modules)
- **Debug an issue** → [Quick Start Guide](QUICK_START_GUIDE.md#troubleshooting)

### By Component

**"I'm working with..."**
- **Configuration management** → Search all docs for "config", "enum", "SuperEnum"
- **Image processing** → Search for "cv", "image", "ImStd"
- **Neural networks** → Search for "model", "ModuleNode", "training"
- **Logging** → Search for "log", "LoggerTemp", "LogTemp"
- **File operations** → Search for "file", "path", "readable", "writable"

## 📱 Quick Reference Cards

### Essential Imports
```python
# Core utilities
from hagike.utils import *

# Logging
from hagike.log import logger_g, LogTemp

# Computer vision
from hagike.basics.cv import *

# Models
from hagike.models.temp import ModuleNode, ModelTemp
from hagike.models.train import TrainerTemp, TrainerInfo
```

### Common Patterns
```python
# Configuration with enums
@advanced_enum()
class Config(SuperEnum):
    learning_rate = 0.001

# Error handling
try:
    # operation
    pass
except Exception as e:
    add_msg(MsgLevel.Error.value, f"Error: {e}")
    error_proc(is_exit=False)

# File safety
if check_path_readable(input_path, is_raise=False):
    ensure_path_writable(output_path)
    # process files
```

## 🔄 Documentation Updates

This documentation was generated by analyzing the codebase structure and examining key modules. It covers:

✅ **Complete API Coverage**: All public functions and classes documented
✅ **Practical Examples**: Real-world usage patterns and code samples  
✅ **Organized Reference**: Easy lookup by module and functionality
✅ **Quick Start Guide**: Get up and running in 5 minutes
✅ **Best Practices**: Framework-specific patterns and recommendations

### Maintenance Notes

- **Auto-generated docs**: Run `./doc.sh` to update HTML documentation
- **Manual docs**: Update these markdown files when adding new APIs
- **Examples**: Keep examples in sync with actual API changes
- **Testing**: Verify examples work with current codebase

## 🤝 Contributing to Documentation

When adding new features to Hagike:

1. **Update docstrings** in the source code
2. **Add examples** to the appropriate documentation files
3. **Update function references** with new APIs
4. **Test examples** to ensure they work
5. **Run documentation generation** with `./doc.sh`

## 📞 Getting Help

- **Documentation Issues**: Check the troubleshooting sections
- **API Questions**: Refer to the function reference and examples
- **Framework Design**: Read the API documentation overview
- **Code Examples**: Look at the `demos/` and `tests/` directories
- **Source Code**: All modules have detailed docstrings

---

**Happy coding with Hagike!** 🚀

*Last updated: Generated from codebase analysis*
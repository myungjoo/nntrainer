# NNtrainer Architecture Analysis

## Overview

NNtrainer is a Software Framework for training Neural Network models on embedded devices with limited resources. The system is designed for on-device AI training and personalization, supporting various machine learning algorithms including Neural Networks, k-NN, Logistic Regression, and Reinforcement Learning.

## Abstract Design View - Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  Applications/                                                  │
│  ├── Android/         ├── Tizen_native/    ├── Custom/         │
│  ├── TransferLearning/ ├── MNIST/          ├── ResNet/         │
│  ├── YOLOv3/          ├── SimpleShot/      ├── LLaMA/          │
│  └── AlexNet/         └── VGG/             └── PicoGPT/        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                           API LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  api/                                                           │
│  ├── capi/           - C API (Public Interface)                │
│  └── ccapi/          - C++ API (Object-Oriented Interface)     │
│                       ↕                                        │
│  nntrainer-api-common.h - Common API Definitions               │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NNTRAINER CORE FRAMEWORK                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Models    │    │    Graph    │    │   Engine    │        │
│  │             │    │             │    │             │        │
│  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │        │
│  │ │neuralnet│ │◄──►│ │network_ │ │◄──►│ │app_     │ │        │
│  │ │        .│ │    │ │graph    │ │    │ │context  │ │        │
│  │ │model_   │ │    │ │graph_   │ │    │ │context  │ │        │
│  │ │loader   │ │    │ │core     │ │    │ │delegate │ │        │
│  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Layers    │    │   Tensor    │    │ Optimizers  │        │
│  │             │    │   System    │    │             │        │
│  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │        │
│  │ │layer_   │ │◄──►│ │tensor   │ │◄──►│ │adam     │ │        │
│  │ │node     │ │    │ │manager  │ │    │ │sgd      │ │        │
│  │ │fc_layer │ │    │ │memory_  │ │    │ │adamw    │ │        │
│  │ │conv2d   │ │    │ │pool     │ │    │ │lr_      │ │        │
│  │ │lstm     │ │    │ │cache_   │ │    │ │scheduler│ │        │
│  │ │...      │ │    │ │pool     │ │    │ └─────────┘ │        │
│  │ └─────────┘ │    │ └─────────┘ │    └─────────────┘        │
│  └─────────────┘    └─────────────┘                           │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Dataset   │    │   Utils     │    │   OpenCL    │        │
│  │             │    │             │    │   Support   │        │
│  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │        │
│  │ │databuffer│◄──►│ │util_func│ │◄──►│ │cl_context│ │        │
│  │ │iteration_│ │    │ │node_    │ │    │ │cl_buffer_│ │        │
│  │ │queue     │ │    │ │exporter │ │    │ │manager  │ │        │
│  │ │data_     │ │    │ │base_    │ │    │ │cl_layers│ │        │
│  │ │producers │ │    │ │properties│ │   │ └─────────┘ │        │
│  │ └─────────┘ │    │ └─────────┘ │    └─────────────┘        │
│  └─────────────┘    └─────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NNSTREAMER INTEGRATION                       │
├─────────────────────────────────────────────────────────────────┤
│  nnstreamer/                                                    │
│  ├── tensor_filter/    - Integration with NNStreamer pipeline  │
│  └── tensor_trainer/   - Training capabilities in pipeline     │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PLATFORM SUPPORT                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Android   │    │    Tizen    │    │   Ubuntu    │        │
│  │     NDK     │    │    6.0M2+   │    │  18.04/20.04│        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Module Relationships

### 1. **API Layer**
- **C API (capi/)**: Public C interface for external applications
- **C++ API (ccapi/)**: Object-oriented C++ interface 
- **Common Definitions**: Shared enums and types in `nntrainer-api-common.h`

**Dependencies**: 
- APIs depend on core framework modules
- Applications interface through APIs exclusively

### 2. **Core Framework Modules**

#### **Models Module**
- **Purpose**: Model lifecycle management, loading/saving, training orchestration
- **Key Components**: 
  - `neuralnet.cpp/h`: Main neural network class
  - `model_loader.cpp/h`: Model serialization/deserialization
- **Dependencies**: Graph, Layers, Optimizers, Dataset

#### **Graph Module** 
- **Purpose**: Computation graph construction and execution management
- **Key Components**:
  - `network_graph.cpp/h`: Graph topology management
  - `graph_core.cpp/h`: Core graph operations
  - `connection.cpp/h`: Node interconnections
- **Dependencies**: Layers (as nodes), Tensor System

#### **Layers Module**
- **Purpose**: Neural network layer implementations
- **Key Components**:
  - **Core Layers**: `fc_layer`, `conv2d_layer`, `lstm`, `gru`
  - **Activation**: Various activation functions
  - **Loss**: `cross_entropy`, `mse_loss`
  - **Preprocessing**: `preprocess_flip`, `preprocess_translate`
  - **OpenCL Layers**: GPU-accelerated implementations in `cl_layers/`
- **Dependencies**: Tensor System, Utils

#### **Tensor System**
- **Purpose**: Multi-precision tensor operations and memory management
- **Key Components**:
  - `tensor.cpp/h`: Core tensor operations
  - `manager.cpp/h`: Memory management
  - `memory_pool.cpp/h`: Memory pooling
  - **Specialized Tensors**: `float_tensor`, `half_tensor`, `int4_tensor`, `uint4_tensor`
  - **Quantization**: `quantizer.cpp/h`, `bcq_tensor`
- **Dependencies**: Utils (for SIMD operations)

#### **Optimizers Module**
- **Purpose**: Training optimization algorithms
- **Key Components**:
  - `adam.cpp/h`: Adam optimizer
  - `sgd.cpp/h`: Stochastic Gradient Descent
  - `adamw.cpp/h`: AdamW optimizer
  - **Learning Rate Schedulers**: Constant, Exponential, Step, Cosine
- **Dependencies**: Tensor System

#### **Dataset Module**
- **Purpose**: Data loading, preprocessing, and batch management
- **Key Components**:
  - `databuffer.cpp/h`: Data buffering
  - `iteration_queue.cpp/h`: Batch iteration management
  - **Data Producers**: Directory, random, file-based producers
- **Dependencies**: Tensor System, Utils

#### **Engine & Context**
- **Purpose**: Application context management and execution control
- **Key Components**:
  - `app_context.cpp/h`: Application-level context
  - `context.h`: Core context management
  - `engine.cpp/h`: Execution engine
- **Dependencies**: All core modules

### 3. **NNStreamer Integration**
- **tensor_filter/**: Enables NNtrainer models to be used in NNStreamer pipelines
- **tensor_trainer/**: Provides training capabilities within streaming pipelines
- **Dependencies**: Core framework, NNStreamer ecosystem

### 4. **Cross-Cutting Concerns**

#### **OpenCL Support**
- **Purpose**: GPU acceleration for supported operations
- **Components**: `cl_context`, `cl_buffer_manager`, `cl_layers/`
- **Integration**: Parallel implementation path for layers and operations

#### **Utilities**
- **Purpose**: Common functionality across modules
- **Components**: 
  - SIMD operations (`util_simd.h`)
  - Node export (`node_exporter.h`)
  - Base properties (`base_properties.h`)
  - Threading (`nntr_threads.h`)

#### **Logging & Error Handling**
- **Components**: `nntrainer_log.h`, `nntrainer_error.h`, `nntrainer_logger.cpp`
- **Integration**: Used throughout all modules

## Data Flow Architecture

```
Input Data → Dataset Module → Tensor System → Graph Module → Layers → Tensor Operations
     ↑                                                                        ↓
Application ← API Layer ← Models Module ← Optimizers ← Gradient Computation ←
```

## Compilation and Build System

- **Build System**: Meson-based with platform-specific configurations
- **Target Platforms**: Android, Tizen, Ubuntu (ARM, x86_64)
- **Language Support**: C/C++ with C++17 standard
- **Dependencies**: Platform-specific OpenCL, TensorFlow Lite (optional)

## Key Design Patterns

1. **Modular Architecture**: Clear separation of concerns with well-defined interfaces
2. **Plugin System**: Support for plugged layers and optimizers
3. **Memory Management**: Sophisticated memory pooling and caching systems
4. **Multi-precision Support**: Various tensor types for different precision requirements
5. **Platform Abstraction**: Unified API across different target platforms
6. **Pipeline Integration**: Seamless integration with NNStreamer streaming framework

This architecture enables NNtrainer to provide efficient on-device neural network training while maintaining flexibility and extensibility across different deployment scenarios.
# NNtrainer Software Architecture Design

## Overview

NNtrainer is an on-device neural network training framework designed for resource-constrained embedded devices. The system provides a complete software stack for training, fine-tuning, and executing neural networks on mobile platforms including Android, Tizen, and desktop systems.

## Abstract Module View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Applications/          │  Tools/               │  Platform Integration    │
│  ├─ MNIST               │  ├─ Android Packaging │  ├─ Android/             │
│  ├─ ResNet              │  ├─ Testing Utils     │  ├─ Tizen_native/        │
│  ├─ VGG                 │  └─ Python Utils      │  └─ Cross-platform       │
│  ├─ YOLO                │                       │                          │
│  ├─ LLaMA               │                       │                          │
│  └─ Custom Examples     │                       │                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│              │                            │                               │
│   C API      │         C++ API            │         JNI                   │
│  (capi/)     │        (ccapi/)            │        (jni/)                 │
│              │                            │                               │
│ ┌─────────┐  │  ┌─────────────────────┐   │  ┌─────────────────────────┐  │
│ │Training │  │  │Object-Oriented      │   │  │Java Native Interface   │  │
│ │Inference│  │  │C++ Bindings         │   │  │Android Integration     │  │
│ │Model    │  │  │Template Support     │   │  │JVM Interoperability    │  │
│ │Config   │  │  │RAII Management      │   │  │Memory Management       │  │
│ └─────────┘  │  └─────────────────────┘   │  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CORE ENGINE LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  App Context    │    │   Core Engine    │    │   Model Management   │   │
│  │  (app_context)  │◄──►│   (engine)       │◄──►│   (models/)          │   │
│  │                 │    │                  │    │                      │   │
│  │ • Configuration │    │ • Execution      │    │ • NeuralNet          │   │
│  │ • Resource Mgmt │    │ • Scheduling     │    │ • Model Loading      │   │
│  │ • Platform Abs  │    │ • Memory Mgmt    │    │ • Serialization      │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────────┘   │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPUTATIONAL GRAPH LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  Network Graph  │    │   Graph Core     │    │   Connections        │   │
│  │ (network_graph) │◄──►│  (graph_core)    │◄──►│  (connection)        │   │
│  │                 │    │                  │    │                      │   │
│  │ • Graph Builder │    │ • Node Mgmt      │    │ • Edge Definition    │   │
│  │ • Topology      │    │ • Execution Flow │    │ • Data Flow          │   │
│  │ • Optimization  │    │ • Memory Plan    │    │ • Dependency Track   │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────────┘   │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NEURAL NETWORK LAYERS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐   │
│ │   Core      │ │  Recurrent  │ │Preprocessing│ │   Loss Functions     │   │
│ │  Layers     │ │   Layers    │ │   Layers    │ │     (loss/)          │   │
│ │             │ │             │ │             │ │                      │   │
│ │• Conv2D     │ │• LSTM       │ │• Flip       │ │• Cross Entropy       │   │
│ │• FC         │ │• GRU        │ │• Translate  │ │• MSE                 │   │
│ │• Pooling    │ │• RNN        │ │• L2Norm     │ │• Custom Loss         │   │
│ │• Dropout    │ │• LSTMCell   │ │             │ │                      │   │
│ │• Activation │ │             │ │             │ │                      │   │
│ └─────────────┘ └─────────────┘ └─────────────┘ └──────────────────────┘   │
│                                                                             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐   │
│ │OpenCL Layers│ │Mathematical │ │Utility      │ │   External Layers    │   │
│ │(cl_layers/) │ │  Operations │ │   Layers    │ │                      │   │
│ │             │ │             │ │             │ │• TensorFlow Lite     │   │
│ │• GPU Accel  │ │• Add/Sub    │ │• Identity   │ │• NNStreamer          │   │
│ │• Memory Opt │ │• Mul/Div    │ │• Reshape    │ │• Plugged Layers      │   │
│ │• Parallel   │ │• Pow/Sqrt   │ │• Split      │ │                      │   │
│ └─────────────┘ └─────────────┘ └─────────────┘ └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPTIMIZATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │   Optimizers    │    │  LR Schedulers   │    │ Training Strategies  │   │
│  │ (optimizers/)   │◄──►│                  │◄──►│                      │   │
│  │                 │    │                  │    │                      │   │
│  │ • Adam          │    │ • Constant       │    │ • Dynamic Training   │   │
│  │ • AdamW         │    │ • Exponential    │    │ • Mixed Precision    │   │
│  │ • SGD           │    │ • Cosine         │    │ • Transfer Learning  │   │
│  │ • Custom Opts   │    │ • Step           │    │ • Fine-tuning        │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TENSOR OPERATIONS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐   │
│ │   Tensor    │ │   Memory    │ │   Data      │ │   Backend Support    │   │
│ │   Core      │ │ Management  │ │   Types     │ │                      │   │
│ │             │ │             │ │             │ │                      │   │
│ │• Operations │ │• Pool Mgmt  │ │• Float32    │ │• CPU Backend         │   │
│ │• Broadcasting│ │• Cache      │ │• Float16    │ │• OpenCL Backend      │   │
│ │• Reshape    │ │• Lazy Load  │ │• Int4/8     │ │• NEON Optimization   │   │
│ │• Arithmetic │ │• Swap       │ │• Quantized  │ │• Memory Mapping      │   │
│ └─────────────┘ └─────────────┘ └─────────────┘ └──────────────────────┘   │
│                                                                             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐   │
│ │Memory       │ │ Optimization│ │Quantization │ │  Specialized Tensors │   │
│ │Planners     │ │  Strategies │ │   Support   │ │                      │   │
│ │             │ │             │ │             │ │• BCQ Tensor          │   │
│ │• Basic      │ │• V1-V3      │ │• Q4K/Q6K    │ │• Character Tensor    │   │
│ │• Optimized  │ │• Memory Opt │ │• Dynamic    │ │• Variable Gradient   │   │
│ │• Cache-aware│ │• Parallel   │ │• Custom     │ │• Weight Management   │   │
│ └─────────────┘ └─────────────┘ └─────────────┘ └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │   Data Buffer   │    │  Data Producers  │    │   Data Iteration     │   │
│  │  (databuffer)   │◄──►│                  │◄──►│                      │   │
│  │                 │    │                  │    │                      │   │
│  │ • Buffer Mgmt   │    │ • File Producers │    │ • Queue Management   │   │
│  │ • Batch Prep    │    │ • Directory Scan │    │ • Async Loading      │   │
│  │ • Preprocessing │    │ • Random Data    │    │ • Threading Support  │   │
│  │ • Augmentation  │    │ • Function Data  │    │ • Memory Efficiency  │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL INTEGRATIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │   NNStreamer    │    │   TensorFlow     │    │   Platform Support   │   │
│  │  Integration    │    │      Lite        │    │                      │   │
│  │                 │    │                  │    │                      │   │
│  │ • Tensor Filter │    │ • Model Import   │    │ • Android NDK        │   │
│  │ • Tensor Train  │    │ • Inference      │    │ • Tizen Native       │   │
│  │ • Pipeline Plug │    │ • Layer Compat   │    │ • Linux/Windows      │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Architectural Principles

### 1. **Layered Architecture**
- Clear separation of concerns with distinct layers
- Each layer provides well-defined interfaces to upper layers
- Abstraction barriers allow for platform-specific implementations

### 2. **Modular Design**
- **Core Engine**: Central execution and orchestration
- **Computational Graph**: Dynamic graph construction and optimization
- **Layer System**: Extensible neural network layer framework
- **Tensor Operations**: High-performance numerical computations
- **Data Pipeline**: Efficient data loading and preprocessing

### 3. **Platform Abstraction**
- Multi-platform support (Android, Tizen, Linux, Windows)
- Hardware acceleration through OpenCL
- Resource-constrained device optimization

### 4. **Extensibility**
- Plugin-based layer system
- Custom optimizer support
- External framework integration (TensorFlow Lite, NNStreamer)

## Module Dependencies and Data Flow

### Primary Data Flow:
1. **Applications** → **API Layer** → **Core Engine**
2. **Core Engine** → **Computational Graph** → **Neural Network Layers**
3. **Layers** → **Tensor Operations** → **Backend Hardware**
4. **Data Pipeline** → **Tensor Operations** → **Layers**
5. **Optimization Layer** → **Layers** (during training)

### Key Interfaces:
- **API Layer**: Provides C, C++, and JNI interfaces
- **Graph Interface**: Manages computational graph construction and execution
- **Layer Interface**: Standardized layer implementation framework
- **Tensor Interface**: High-performance tensor operations
- **Backend Interface**: Hardware abstraction for CPU/GPU execution

## Memory Management Strategy

### Multi-tier Memory System:
1. **Memory Pools**: Efficient allocation/deallocation
2. **Memory Planners**: Optimization strategies (Basic, V1-V3)
3. **Cache Management**: Layer-wise caching and lazy loading
4. **Swap Devices**: Virtual memory support for large models

### Quantization Support:
- **Multiple Precision**: FP32, FP16, INT4, INT8
- **Dynamic Quantization**: Runtime precision adjustment
- **Specialized Tensors**: BCQ, Q4K, Q6K formats

## Performance Optimizations

### Computational Optimizations:
- **OpenCL Acceleration**: GPU-accelerated layers
- **NEON SIMD**: ARM processor optimization
- **Memory Access Patterns**: Cache-friendly algorithms
- **Parallel Execution**: Multi-threaded operations

### Training Optimizations:
- **Mixed Precision Training**: Reduced memory footprint
- **Dynamic Training**: Adaptive resource allocation
- **Transfer Learning**: Efficient fine-tuning workflows

## Extension Points

### 1. **Custom Layers**
- Implement `layer_devel.h` interface
- Plugin-based registration system
- Support for both CPU and GPU implementations

### 2. **Custom Optimizers**
- Extend `optimizer_wrapped` base class
- Learning rate scheduler integration
- Custom gradient computation

### 3. **External Framework Integration**
- TensorFlow Lite layer wrapper
- NNStreamer pipeline integration
- Custom tensor filter implementations

## Quality Attributes

### Performance:
- Optimized for mobile/embedded devices
- Memory-efficient tensor operations
- Hardware acceleration support

### Portability:
- Cross-platform compatibility
- Abstracted hardware interfaces
- Build system flexibility (Meson)

### Extensibility:
- Plugin architecture
- Well-defined extension points
- Modular component design

### Maintainability:
- Clear module boundaries
- Comprehensive testing framework
- Documentation and examples

This architecture enables NNtrainer to provide a comprehensive on-device training solution while maintaining efficiency, flexibility, and ease of use across different platforms and hardware configurations.
# NNtrainer Software Architecture Analysis

## Overview

NNtrainer is a comprehensive Software Framework for training Neural Network models on embedded devices with limited resources. It provides on-device training capabilities with support for multiple hardware backends and various neural network architectures.

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              NNtrainer Framework                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                  Application Layer                              │
├─────────────────────┬───────────────────┬─────────────────────┬─────────────────┤
│      C API         │      C++ API      │   NNStreamer       │   Applications  │
│    (capi/)         │     (ccapi/)      │  Integration       │  (Examples &    │
│                    │                   │   (nnstreamer/)    │   Use Cases)    │
└─────────────────────┴───────────────────┴─────────────────────┴─────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 Core Framework                                  │
├─────────────────────┬───────────────────┬─────────────────────┬─────────────────┤
│    Model Layer     │   Graph Layer     │   Compiler Layer   │   Engine Layer  │
│   (models/)        │   (graph/)        │   (compiler/)       │   (engine.h)    │
│                    │                   │                     │                 │
│ • NeuralNet        │ • NetworkGraph    │ • Multi-format      │ • Multi-backend │
│ • Model Loader     │ • Graph Core      │   Support           │   Management    │
│ • Dynamic Training │ • Node Management │ • TFLite, ONNX,     │ • CPU, GPU, NPU │
│   Optimization     │ • Connections     │   INI Interpreters  │ • Context Mgmt  │
└─────────────────────┴───────────────────┴─────────────────────┴─────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Computation Layer                                  │
├─────────────────────┬───────────────────┬─────────────────────┬─────────────────┤
│     Layers         │    Optimizers     │      Loss           │     Tensor      │
│   (layers/)        │  (optimizers/)    │   (layers/loss/)    │   (tensor/)     │
│                    │                   │                     │                 │
│ • Dense/FC         │ • Adam/AdamW      │ • Cross Entropy     │ • Memory Mgmt   │
│ • Convolution      │ • SGD             │ • MSE Loss          │ • Quantization  │
│ • LSTM/GRU/RNN     │ • Learning Rate   │ • KLD Loss          │ • Multiple      │
│ • Activation       │   Schedulers      │ • Custom Loss       │   Data Types    │
│ • Normalization    │                   │                     │ • SIMD Ops      │
└─────────────────────┴───────────────────┴─────────────────────┴─────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               Backend Layer                                     │
├─────────────────────┬───────────────────┬─────────────────────┬─────────────────┤
│   CPU Backend      │   OpenCL (GPU)    │    Data Pipeline    │    Utilities    │
│ (tensor/cpu_backend)│   (opencl/)       │    (dataset/)       │    (utils/)     │
│                    │                   │                     │                 │
│ • NEON/SIMD        │ • GPU Kernels     │ • Data Producers    │ • Profiling     │
│ • Optimized Ops    │ • Buffer Mgmt     │ • Data Loaders      │ • Threading     │
│ • Memory Planning  │ • Command Queues  │ • Iteration Queues  │ • INI Config    │
│                    │ • Context Mgmt    │ • Random/File Data  │ • Logging       │
└─────────────────────┴───────────────────┴─────────────────────┴─────────────────┘
```

## Module Dependency Diagram

```
                                    Applications
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   API Layer     │
                              │  ┌───────────┐  │
                              │  │ C API     │  │
                              │  │ C++ API   │  │
                              │  │ NNStream  │  │
                              │  └───────────┘  │
                              └─────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │  Model Layer    │
                              │  ┌───────────┐  │
                              │  │ NeuralNet │◄─┼─── Model Loader
                              │  │ Training  │  │
                              │  │ Optimizer │  │
                              │  └───────────┘  │
                              └─────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │          Graph Layer                │
                     │  ┌─────────────┐  ┌─────────────┐   │
                     │  │ Network     │  │ Compiler    │   │
                     │  │ Graph       │  │ System      │   │
                     │  │ ┌─────────┐ │  │ ┌─────────┐ │   │
                     │  │ │ Nodes   │ │  │ │ TFLite  │ │   │
                     │  │ │ Edges   │ │  │ │ ONNX    │ │   │
                     │  │ └─────────┘ │  │ │ INI     │ │   │
                     │  └─────────────┘  │ └─────────┘ │   │
                     └─────────────────────────────────────┘
                                        │
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                    Computation Layer                         │
        │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐     │
        │  │ Layers  │  │Optimizer│  │  Loss   │  │   Tensor    │     │
        │  │         │  │         │  │         │  │             │     │
        │  │• Dense  │  │• Adam   │  │• Cross  │  │• Memory     │     │
        │  │• Conv   │  │• SGD    │  │  Entropy│  │  Pool       │     │
        │  │• LSTM   │  │• LR     │  │• MSE    │  │• Quantize   │     │
        │  │• RNN    │  │  Sched  │  │• KLD    │  │• Data Types │     │
        │  │• Activ  │  │         │  │         │  │• Operations │     │
        │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘     │
        └──────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                      Backend Layer                           │
        │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐     │
        │  │   CPU   │  │ OpenCL  │  │ Dataset │  │  Utilities  │     │
        │  │ Backend │  │ (GPU)   │  │Pipeline │  │             │     │
        │  │         │  │         │  │         │  │• Profiler   │     │
        │  │• SIMD   │  │• Kernels│  │• Loaders│  │• Threading  │     │
        │  │• Memory │  │• Buffers│  │• Queues │  │• Config     │     │
        │  │  Plan   │  │• Context│  │• Random │  │• Logging    │     │
        │  │         │  │         │  │• File   │  │             │     │
        │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘     │
        └──────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │  Engine Layer   │
                              │                 │
                              │ • Context Mgmt  │
                              │ • Backend Route │
                              │ • Resource Mgmt │
                              │ • Multi-HW Sup  │
                              └─────────────────┘
```

## Key Architectural Components

### 1. **Application Interface Layer**
- **C API (capi/)**: Pure C interface for maximum compatibility
- **C++ API (ccapi/)**: Modern C++ interface with advanced features
- **NNStreamer Integration**: Streaming ML pipeline integration
- **Applications**: Example implementations and use cases

### 2. **Model Management Layer**
- **NeuralNet**: High-level neural network model interface
- **Model Loader**: Support for loading pre-trained models
- **Dynamic Training Optimization**: Runtime optimization strategies
- **Model Properties**: Configuration and metadata management

### 3. **Graph Computation Layer**
- **Network Graph**: Computational graph representation and execution
- **Graph Core**: Core graph algorithms and optimizations
- **Connection Management**: Node connectivity and data flow
- **Compiler System**: Multi-format model compilation (TFLite, ONNX, INI)

### 4. **Computation Primitives Layer**
- **Layers**: Extensive neural network layer implementations
  - Dense/Fully Connected, Convolution (1D/2D), LSTM/GRU/RNN
  - Activation functions, Normalization, Pooling, etc.
- **Optimizers**: Training optimization algorithms (Adam, SGD, etc.)
- **Loss Functions**: Various loss implementations
- **Tensor Operations**: Core tensor mathematics and operations

### 5. **Backend Execution Layer**
- **CPU Backend**: Optimized CPU implementations with SIMD
- **OpenCL Backend**: GPU acceleration support
- **Dataset Pipeline**: Data loading and preprocessing
- **Memory Management**: Efficient memory allocation and pooling

### 6. **Engine Management Layer**
- **Context Management**: Multi-backend execution context
- **Resource Allocation**: Hardware resource management
- **Backend Routing**: Automatic backend selection
- **Cross-platform Support**: Android, Tizen, Linux support

## Data Flow Architecture

```
Input Data → Dataset Pipeline → Tensor Operations → Layer Execution
     ▲              │                    │               │
     │              ▼                    ▼               ▼
Model Config → Data Loaders → Memory Pool → Graph Execution
     │              │                    │               │
     ▼              ▼                    ▼               ▼
Compiler → Iteration Queue → Backend Selection → Optimization
     │              │                    │               │
     ▼              ▼                    ▼               ▼
Graph IR → Training Loop → Hardware Context → Model Update
```

## Key Design Patterns

1. **Strategy Pattern**: Multiple backend implementations (CPU, OpenCL)
2. **Factory Pattern**: Layer creation, optimizer instantiation
3. **Observer Pattern**: Training callbacks and monitoring
4. **Builder Pattern**: Model construction and configuration
5. **Bridge Pattern**: API abstraction over core implementation
6. **Plugin Architecture**: Extensible layer and optimizer system

## Hardware Acceleration Strategy

```
┌─────────────────────────────────────────────────────────┐
│                   Engine Manager                        │
├─────────────────────────────────────────────────────────┤
│           Context Selection Logic                       │
├─────────────┬─────────────┬─────────────┬─────────────┤
│     CPU     │   OpenCL    │     NPU     │   Future    │
│   Context   │   Context   │   Context   │   Backends  │
│             │             │             │             │
│ • SIMD Ops  │ • GPU       │ • Hardware  │ • CUDA      │
│ • Threading │   Kernels   │   Specific  │ • Vulkan    │
│ • Memory    │ • Buffers   │   Accel     │ • Custom    │
│   Planning  │ • Queues    │             │   Hardware  │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

## Memory Management Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Memory Manager                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Memory    │  │   Tensor    │  │ Quantization│     │
│  │    Pool     │  │    Pool     │  │   Support   │     │
│  │             │  │             │  │             │     │
│  │• Allocation │  │• Lifecycle  │  │• FP16       │     │
│  │• Reuse      │  │• Caching    │  │• INT4/8     │     │
│  │• Planning   │  │• Swapping   │  │• Custom     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Platform Support Matrix

| Platform | Architecture | API Support | Acceleration |
|----------|-------------|-------------|--------------|
| Android  | ARM/ARM64   | C/C++/JNI   | CPU, GPU     |
| Tizen    | ARM/x86     | C/C++       | CPU, GPU     |
| Linux    | x86_64      | C/C++       | CPU, GPU     |
| Windows  | x86_64      | C/C++       | CPU          |

## Summary

NNtrainer employs a layered, modular architecture that supports:

- **Multi-backend execution** with automatic hardware selection
- **Comprehensive neural network support** with extensive layer library
- **Memory-efficient operations** optimized for mobile/embedded devices
- **Cross-platform compatibility** across major mobile and desktop platforms
- **Extensible design** allowing custom layers, optimizers, and backends
- **Multiple model format support** for interoperability
- **On-device training capabilities** with resource-aware optimizations

The architecture emphasizes modularity, performance, and resource efficiency, making it suitable for deployment on resource-constrained devices while maintaining the flexibility to leverage available hardware acceleration.
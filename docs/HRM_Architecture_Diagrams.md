# HRM System Architecture Diagrams

## 1. Complete System Architecture (Mermaid)

```mermaid
graph TB
    subgraph "Data Sources"
        DS1[Security Dataset<br/>6,600 records]
        DS2[SWE Dataset<br/>9,000 records]
        DS3[SQE Dataset<br/>7,200 records]
        DS4[Reasoning Dataset<br/>4,000 records]
    end

    subgraph "Data Integration Pipeline"
        subgraph "Parsing Stage"
            P1[JSONLParser<br/>Security]
            P2[JSONLParser<br/>SWE]
            P3[JSONLParser<br/>SQE]
            P4[JSONLParser<br/>Reasoning]
        end
        
        subgraph "Transformation Stage"
            T1[CharacterTokenizer<br/>Vocab: 65,536]
            T2[HRMDataTransformer<br/>Max Length: 512]
            T3[Puzzle ID Mapping]
        end
        
        subgraph "Validation Stage"
            V1[EmptyFilter]
            V2[DuplicateFilter]
            V3[LengthValidator]
            V4[CompletenessValidator]
            V5[QualityValidator]
        end
        
        subgraph "Output Stage"
            O1[Train Split<br/>80%]
            O2[Val Split<br/>10%]
            O3[Test Split<br/>10%]
        end
    end

    subgraph "SQE-Enhanced HRM Model"
        subgraph "Base HRM"
            HRM1[Token Embedding<br/>vocab_size × hidden_size]
            HRM2[Rotary Position<br/>Encoding RoPE]
            HRM3[Attention Layers<br/>Multi-head + Flash/SDPA]
            HRM4[Feed Forward<br/>SwiGLU Activation]
            HRM5[LM Head<br/>Language Modeling]
        end
        
        subgraph "SQE Components"
            SQE1[SQE Projection<br/>hidden → hidden]
            SQE2[SQE LayerNorm]
            SQE3[SQE Attention<br/>Stacked Layers]
            SQE4[Gradient Preserving<br/>Wrapper]
        end
    end

    subgraph "Training System"
        TR1[Data Loaders<br/>Batching + Padding]
        TR2[Forward Pass<br/>Carry State]
        TR3[Loss Computation<br/>Cross Entropy]
        TR4[Backward Pass<br/>Gradient Flow]
        TR5[Optimizer<br/>AdamW]
        TR6[Checkpoint<br/>Management]
    end

    subgraph "Monitoring"
        M1[Weights & Biases<br/>Metrics Tracking]
        M2[Pipeline Metrics<br/>JSON Logs]
        M3[Model Checkpoints<br/>Best Model]
    end

    %% Data Flow
    DS1 --> P1
    DS2 --> P2
    DS3 --> P3
    DS4 --> P4

    P1 --> T1
    P2 --> T1
    P3 --> T1
    P4 --> T1

    T1 --> T2
    T2 --> T3

    T3 --> V1
    V1 --> V2
    V2 --> V3
    V3 --> V4
    V4 --> V5

    V5 --> O1
    V5 --> O2
    V5 --> O3

    O1 --> TR1
    O2 --> TR1
    O3 --> TR1

    TR1 --> HRM1
    HRM1 --> HRM2
    HRM2 --> HRM3
    HRM3 --> HRM4
    HRM4 --> HRM5

    HRM5 --> SQE1
    SQE1 --> SQE2
    SQE2 --> SQE3
    SQE3 --> SQE4

    SQE4 --> TR2
    TR2 --> TR3
    TR3 --> TR4
    TR4 --> TR5
    TR5 --> TR6

    TR6 --> M3
    TR3 --> M1
    V5 --> M2

    style DS1 fill:#e1f5fe
    style DS2 fill:#e1f5fe
    style DS3 fill:#e1f5fe
    style DS4 fill:#e1f5fe
    style HRM1 fill:#f3e5f5
    style SQE1 fill:#fff9c4
    style TR1 fill:#e8f5e9
    style M1 fill:#fce4ec
```

## 2. Data Pipeline Flow (Mermaid)

```mermaid
flowchart LR
    subgraph Input["📥 Input Data"]
        A1[JSONL Files<br/>26,800 records]
    end

    subgraph Parse["🔍 Parse"]
        B1{Field Mapping}
        B2[Extract prompt/completion]
        B3[Extract metadata]
    end

    subgraph Transform["🔄 Transform"]
        C1[Tokenize Text<br/>Character Level]
        C2[Add Special Tokens<br/>BOS/EOS/PAD]
        C3[Truncate/Pad<br/>Max Length: 512]
        C4[Assign Puzzle IDs<br/>Hash Strategy]
    end

    subgraph Validate["✅ Validate"]
        D1{Length Check<br/>10-100k chars}
        D2{Quality Score<br/>> 0.7}
        D3{Completeness<br/>> 0.8}
        D4{Duplicate Check}
    end

    subgraph Output["📤 Output"]
        E1[Train: 56 examples<br/>163KB]
        E2[Val: 7 examples<br/>20KB]
        E3[Test: 7 examples<br/>20KB]
    end

    A1 --> B1
    B1 -->|Found| B2
    B1 -->|Not Found| B1
    B2 --> B3
    B3 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> D1
    D1 -->|Pass| D2
    D1 -->|Fail| X1[❌ Discard]
    D2 -->|Pass| D3
    D2 -->|Fail| X1
    D3 -->|Pass| D4
    D3 -->|Fail| X1
    D4 -->|Unique| E1
    D4 -->|Duplicate| X2[❌ Remove<br/>22,730 items]
    
    E1 -.80%.-> F[Training]
    E2 -.10%.-> F
    E3 -.10%.-> F

    style A1 fill:#bbdefb
    style E1 fill:#c8e6c9
    style E2 fill:#fff9c4
    style E3 fill:#ffccbc
    style X1 fill:#ffcdd2
    style X2 fill:#ffcdd2
```

## 3. HRM Model Architecture (Mermaid)

```mermaid
graph TD
    subgraph Input["Input Processing"]
        I1[Token IDs<br/>batch × seq_len]
        I2[Puzzle IDs<br/>batch × 1]
        I3[Target IDs<br/>batch × seq_len]
    end

    subgraph Embedding["Embedding Layer"]
        E1[Token Embedding<br/>65536 → 768]
        E2[Position Embedding<br/>RoPE]
    end

    subgraph Core["HRM Core Layers"]
        C1[Layer 1<br/>Attention + FFN]
        C2[Layer 2<br/>Attention + FFN]
        C3[Layer N<br/>Attention + FFN]
    end

    subgraph Attention["Attention Mechanism"]
        A1{Flash Attention<br/>Available?}
        A2[Flash Attention<br/>GPU Optimized]
        A3[SDPA Fallback<br/>CPU/Windows]
    end

    subgraph FFN["Feed Forward Network"]
        F1[Linear 1<br/>768 → 3072]
        F2[SwiGLU<br/>Activation]
        F3[Linear 2<br/>3072 → 768]
    end

    subgraph SQE["SQE Enhancement"]
        S1[SQE Projection<br/>768 → 768]
        S2[SQE LayerNorm]
        S3[Multi-head Attention<br/>Stacked × 2]
        S4[Enhanced Representation]
    end

    subgraph Output["Output Processing"]
        O1[LM Head<br/>768 → 65536]
        O2[Logits<br/>batch × seq_len × vocab]
        O3[Enhanced Logits<br/>+ 0.1 × enhancement]
    end

    I1 --> E1
    E1 --> E2
    I2 --> E2
    E2 --> C1
    C1 --> C2
    C2 --> C3
    
    C1 -.contains.-> A1
    A1 -->|Yes| A2
    A1 -->|No| A3
    
    C1 -.contains.-> F1
    F1 --> F2
    F2 --> F3
    
    C3 --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> O1
    O1 --> O2
    S4 -.enhancement.-> O3
    O3 -.combined.-> O2

    I3 -.loss computation.-> O2

    style I1 fill:#e3f2fd
    style S1 fill:#fff9c4
    style O2 fill:#f3e5f5
    style A2 fill:#c8e6c9
    style A3 fill:#ffccbc
```

## 4. Training Loop (Mermaid)

```mermaid
stateDiagram-v2
    [*] --> LoadData: Initialize

    LoadData --> CreateLoaders: Train/Val/Test Splits
    CreateLoaders --> InitModel: Batch Size = 4
    InitModel --> InitOptimizer: SQE-Enhanced HRM
    InitOptimizer --> TrainEpoch: AdamW, LR=1e-4

    state TrainEpoch {
        [*] --> LoadBatch
        LoadBatch --> InitCarry: Get Next Batch
        InitCarry --> Forward: Initialize State
        Forward --> ComputeLoss: Model(carry, batch)
        ComputeLoss --> Backward: Cross Entropy
        Backward --> UpdateWeights: loss.backward()
        UpdateWeights --> NextBatch: optimizer.step()
        NextBatch --> LoadBatch: More Batches?
        NextBatch --> [*]: Epoch Done
    }

    TrainEpoch --> Validate: After Each Epoch
    
    state Validate {
        [*] --> EvalMode
        EvalMode --> ValBatch: model.eval()
        ValBatch --> ValForward: No Gradients
        ValForward --> ValLoss: Compute Loss
        ValLoss --> NextVal: Accumulate
        NextVal --> ValBatch: More Batches?
        NextVal --> [*]: Val Done
    }

    Validate --> CheckImprovement: Compare Val Loss
    CheckImprovement --> SaveCheckpoint: Loss Improved?
    SaveCheckpoint --> LogMetrics: Save Model
    LogMetrics --> NextEpoch: W&B Logging
    NextEpoch --> TrainEpoch: More Epochs?
    NextEpoch --> [*]: Training Complete

    note right of LoadData
        Processed Data:
        - Train: 56 examples
        - Val: 7 examples
        - Test: 7 examples
    end note

    note right of InitModel
        Model Config:
        - Hidden: 768
        - Heads: 12
        - Layers: 12
        - SQE Layers: 2
    end note

    note right of SaveCheckpoint
        Checkpoint Contains:
        - Model State Dict
        - Optimizer State
        - Epoch Number
        - Validation Loss
    end note
```

## 5. Component Interaction (Mermaid)

```mermaid
sequenceDiagram
    participant User
    participant Pipeline as Data Pipeline
    participant Parser as JSONL Parser
    participant Transformer as HRM Transformer
    participant Validator as Data Validator
    participant Storage as File Storage
    participant Trainer as Training Script
    participant Model as SQE-HRM Model
    participant WandB as Weights & Biases

    User->>Pipeline: Run with config
    Pipeline->>Parser: Parse 26,800 records
    Parser->>Parser: Extract fields
    Parser-->>Pipeline: 22,800 valid records
    
    Pipeline->>Transformer: Transform records
    Transformer->>Transformer: Tokenize text
    Transformer->>Transformer: Assign puzzle IDs
    Transformer-->>Pipeline: 22,800 HRM examples
    
    Pipeline->>Validator: Validate examples
    Validator->>Validator: Check quality (>0.7)
    Validator->>Validator: Remove duplicates
    Validator-->>Pipeline: 70 unique examples
    
    Pipeline->>Storage: Save splits
    Storage-->>Pipeline: Train/Val/Test saved
    Pipeline-->>User: Pipeline complete
    
    User->>Trainer: Start training
    Trainer->>Storage: Load datasets
    Storage-->>Trainer: 56/7/7 examples
    
    Trainer->>Model: Initialize SQE-HRM
    Model-->>Trainer: Model ready
    
    loop Each Epoch
        Trainer->>Model: Forward pass
        Model->>Model: Compute embeddings
        Model->>Model: Apply attention
        Model->>Model: SQE enhancement
        Model-->>Trainer: Logits
        
        Trainer->>Trainer: Compute loss
        Trainer->>Model: Backward pass
        Model-->>Trainer: Gradients
        
        Trainer->>Trainer: Update weights
        Trainer->>WandB: Log metrics
        
        Trainer->>Model: Validate
        Model-->>Trainer: Val loss
        
        alt Loss Improved
            Trainer->>Storage: Save checkpoint
        end
    end
    
    Trainer-->>User: Training complete
```

## Metrics Summary

### Data Pipeline Metrics
- **Input Records**: 26,800
- **Parsed Successfully**: 22,800 (85%)
- **After Deduplication**: 70 unique (0.3%)
- **Processing Time**: 11.23s
- **Train/Val/Test Split**: 56/7/7

### Model Architecture
- **Embedding Size**: 768
- **Attention Heads**: 12
- **Layers**: 12
- **Vocabulary**: 65,536 tokens
- **Sequence Length**: 512
- **SQE Layers**: 2
- **Parameters**: ~85M (base) + SQE

### Training Configuration
- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Precision**: BFloat16 (GPU) / FP32 (CPU)
- **Device**: CUDA or CPU


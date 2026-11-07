# Model Card: Qwen2.5-0.5B-Instruct

## Model Overview
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Developer**: Alibaba Cloud
- **Parameters**: 490 Million
- **Role in AtelierAI**: Natural language interpretation into strict Pydantic JSON schema.

## Usage & Prompting Strategy
- **Zero-Shot Prompting**: Defines Fashion-MNIST taxonomy and schema constraints directly in system context.
- **Few-Shot Prompting**: Provides exemplar tuples of user requests and target JSON outputs to guide ambiguous phrasing and plural count parsing.
- **Schema**:
  ```json
  {
    "class_label": "ankle boot",
    "count": 4
  }
  ```
- **Execution**: Evaluated locally using Hugging Face Transformers with fp16 acceleration.

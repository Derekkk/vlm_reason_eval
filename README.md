# VLM Reasoning Evaluation

A comprehensive evaluation framework for Vision-Language Models (VLMs) on mathematical reasoning tasks. This project provides tools to evaluate Qwen2.5-VL and other VLM models on multiple math and vision datasets with support for pass@k metrics, multi-GPU inference, and batch processing.

## Features

- 🎯 **Multi-Dataset Support**: Evaluate on 12+ datasets including MathVista, MathVerse, MathVision, EMMA, MMMU-Pro, and more
- 📊 **Advanced Metrics**: Calculate accuracy, pass@k, majority vote, and mean accuracy
- 🚀 **Batch Processing**: Generate multiple answers per question for robust evaluation
- 💻 **Multi-GPU Support**: Automatic model distribution across multiple GPUs using `device_map="auto"`
- 🔄 **Batch Inference**: Efficient batch processing for non-vLLM models
- 📝 **Comprehensive Logging**: Detailed logs with timestamps for each evaluation run
- 🎛️ **Flexible Configuration**: Easy-to-configure model and dataset settings

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| **MathVista** | Mathematical reasoning with visual elements |
| **MathVerse** | Multi-modal mathematical problem solving |
| **MathVision** | Vision-based math problems |
| **HallusionBench** | Hallucination detection benchmark |
| **EMMA-Math** | Mathematical problems from EMMA |
| **EMMA-Chem** | Chemistry problems from EMMA |
| **EMMA-Code** | Coding problems from EMMA |
| **EMMA-Physics** | Physics problems from EMMA |
| **MMMU-Pro-10** | MMMU Pro with 10 options |
| **MMMU-Pro-4** | MMMU Pro with 4 options |
| **MMMU-Pro-Vision** | Vision-only MMMU Pro |
| **SFTSeed** | Fine-tuning seed dataset |

## Project Structure

```
vlm_reason_eval/
├── main.py              # Main evaluation script
├── run_eval.sh          # Batch evaluation script for multiple models
├── utils/
│   ├── config.py        # Dataset and model configuration
│   ├── data_utils.py     # Data loading and processing utilities
│   ├── model_utils.py    # Model loading and generation utilities
│   └── metric_utils.py   # Metric calculation (accuracy, pass@k)
├── outputs/             # Evaluation results (JSON files)
├── logs/                # Evaluation logs
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU(s)
- PyTorch
- Transformers library

### Dependencies

```bash
pip install torch transformers datasets qwen-vl-utils mathruler tqdm
```

Optional (for vLLM support):
```bash
pip install vllm
```

## Quick Start

### 1. Single Model Evaluation

```bash
python main.py \
    --model_path /path/to/model \
    --dataset mathvista mathverse \
    --n 10 \
    --data_subset 100
```

### 2. Batch Evaluation (Multiple Models)

Edit `run_eval.sh` to add your models:

```bash
MODEL_PATHS=(
    "/path/to/model1"
    "/path/to/model2"
    "/path/to/model3"
)
```

Then run:
```bash
bash run_eval.sh
```

## Usage

### Command Line Arguments

```bash
python main.py --help
```

**Main Arguments:**
- `--model_path`: Path to the model directory
- `--dataset`: One or more datasets to evaluate (space-separated)
- `--n`: Number of answers to generate per question (for pass@k calculation)
- `--data_subset`: Number of samples to evaluate (-1 for all)

### Evaluation Metrics

The framework calculates several metrics:

1. **First Answer Accuracy**: Accuracy using only the first generated answer
2. **Pass@k**: Percentage of questions where at least one of the first k answers is correct
3. **Majority Vote Accuracy**: Accuracy using majority voting across all generated answers
4. **Mean Accuracy**: Average accuracy across all generated answers

### Example Output

```
Completed mathvista! Final accuracy: 0.4500 (45/100)
  Pass@1: 0.4500 (45/100)
  Pass@5: 0.6200 (62/100)
  Pass@10: 0.6800 (68/100)
  Majority Vote: 0.6500 (65/100)
  Mean Accuracy: 0.5200
```

## Configuration

### Model Configuration

Models are configured in `utils/config.py`. You can adjust:

- `max_new_tokens`: Maximum tokens to generate (default: 1000)
- `temperature`: Sampling temperature (default: 1.0)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `top_k`: Top-k sampling parameter (default: 50)

### Dataset Configuration

Each dataset has its own configuration in `utils/config.py`:

- Image field mapping
- Instruction/query field
- Response/answer field
- Options/choices handling

## Multi-GPU Inference

The framework automatically supports multi-GPU inference:

- **Transformers Mode**: Uses `device_map="auto"` to distribute model across all available GPUs
- **vLLM Mode**: Automatically uses tensor parallelism across all GPUs

Set `CUDA_VISIBLE_DEVICES` to control which GPUs to use:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --model_path /path/to/model
```

## Batch Processing

The framework supports efficient batch processing:

- **Multiple Answers**: Generate n answers per question in a single batch
- **Batch Inference**: Process multiple samples simultaneously
- **Memory Efficient**: Optimized for large-scale evaluation

## Output Format

Results are saved in JSON format in the `outputs/` directory:

```json
{
  "instruction": "Question text",
  "response": "Ground truth answer",
  "reasoning": "Generated reasoning",
  "answer": "Generated answer",
  "correct": 1,
  "n_answers": 10,
  "all_answers": ["answer1", "answer2", ...],
  "all_reasonings": ["reasoning1", "reasoning2", ...],
  "all_correct": [true, false, true, ...]
}
```

## Logging

Evaluation logs are saved in the `logs/` directory with timestamps:

- Format: `log-{model_name}-{timestamp}`
- Includes: Configuration, progress, metrics, and errors
- Can be viewed in real-time with `tee` command

## Advanced Usage

### Custom System Prompts

Modify the system prompt in `main.py` or pass it as a parameter:

```python
SYSTEM_PROMPT = """Your custom system prompt here."""
```

### Per-Model Configuration

In `run_eval.sh`, you can configure different settings per model:

```bash
get_is_reason() {
    local model_path="$1"
    case "$model_path" in
        "/path/to/model1")
            echo "False"
            ;;
        *)
            echo "True"
            ;;
    esac
}
```

## Troubleshooting

### Out of Memory

- Reduce `--data_subset` to evaluate fewer samples
- Reduce `--n` to generate fewer answers per question
- Use fewer GPUs by setting `CUDA_VISIBLE_DEVICES`

### Model Loading Issues

- Ensure model path is correct and accessible
- Check that model format is compatible (HuggingFace format)
- Verify GPU memory is sufficient

### Dataset Loading Errors

- Check internet connection for HuggingFace datasets
- Verify dataset names are correct
- Ensure sufficient disk space for dataset caching

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Add your license here]

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
[Add citation information]
```

## Acknowledgments

- Qwen2.5-VL team for the base model
- Dataset creators for providing evaluation benchmarks
- Open-source community for tools and libraries

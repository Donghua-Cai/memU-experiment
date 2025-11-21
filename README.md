# LoCoMo Memory Test System

A comprehensive memory testing framework for evaluating conversational memory agents using the LoCoMo (Long-term Conversational Memory) benchmark.

## Overview

This system evaluates an AI agent's ability to:
1. **Process conversational sessions** - Extract and store character memories from multi-turn conversations
2. **Answer questions** - Respond to questions about the conversations using stored memories
3. **Evaluate accuracy** - Assess answer correctness across different question categories
4. **Analyze errors** - Generate detailed error reports for incorrect answers

The system uses specialized agents:
- **MemAgent**: Manages memory creation and updates from conversation sessions
- **ResponseAgent**: Retrieves relevant memories and generates answers to questions
- **EvaluateAgent**: Evaluates answer accuracy and performs error analysis

## Requirements

### Environment
- Python 3.8+
- OpenAI API access

### Dependencies
```bash
pip install openai python-dotenv
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom OpenAI endpoint
# OPENAI_BASE_URL=https://your-custom-endpoint.com/v1
```

### 2. Data Preparation

Place your LoCoMo test data file in the `data/` directory:
```
data/locomo10.json
```

The data file should contain conversation sessions and QA pairs in the LoCoMo format.

## Usage

### Basic Usage

Run the test with default settings:
```bash
python locomo_test.py
```

### Common Examples

**Test with specific model:**
```bash
python locomo_test.py --chat-deployment gpt-4o
```

**Test first 3 samples only:**
```bash
python locomo_test.py --sample-use "3"
```

**Test specific samples by index:**
```bash
python locomo_test.py --sample-use "[0, 2, 5]"
```

**Filter by question category:**
```bash
python locomo_test.py --category "1,2,3"
```

**Force regenerate memories:**
```bash
python locomo_test.py --force-resum
```

**Use profile information in answers:**
```bash
python locomo_test.py --use-profile search
```

**Skip evaluation (only process sessions):**
```bash
python locomo_test.py --no-eval
```

**Analyze all questions (not just errors):**
```bash
python locomo_test.py --analyze-on all
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-file` | str | `data/locomo10.json` | Path to test data file |
| `--sample-use` | str | None (all) | Sample indices: number (e.g., "5") or list (e.g., "[0,1,3]") |
| `--memory-dir` | str | `memory` | Directory for storing memory files |
| `--chat-deployment` | str | `gpt-4o-mini` | OpenAI model name |
| `--max-workers` | int | 5 | Number of parallel workers for QA processing |
| `--category` | str | None (all) | Question categories to test (e.g., "1" or "0,2,3") |
| `--use-image` | bool | True | Include image captions in conversations |
| `--use-profile` | str | `none` | Profile usage: "none", "search", or "all" |
| `--force-resum` | flag | False | Force regenerate all memories from scratch |
| `--no-eval` | flag | False | Skip evaluation (only process sessions) |
| `--analyze-on` | str | `wrong` | Detailed analysis: "all", "wrong", or "none" |

### Parameter Details

#### `--sample-use`
Controls which samples to process:
- **Integer**: `"5"` - Process first 5 samples
- **List**: `"[0, 2, 5, 8]"` - Process samples at specific indices
- **Omit**: Process all samples

#### `--category`
Filter questions by category:
- **Single**: `"1"` - Only category 1 questions
- **Multiple**: `"0,2,3"` - Categories 0, 2, and 3
- **Omit**: All categories

#### `--use-profile`
Control profile information usage:
- `none` - Don't use profile information (default)
- `search` - Search profiles when relevant
- `all` - Always include profile information

#### `--analyze-on`
Control detailed error analysis:
- `wrong` - Analyze only incorrect answers (default)
- `all` - Analyze all answers
- `none` - Skip detailed analysis

## Output Files

The system generates several output files:

### 1. Test Results
**File**: `enhanced_memory_test_results_TIMESTAMP.json`

Contains:
- Complete test configuration and arguments
- Summary statistics (accuracy, timing, etc.)
- Category-wise performance breakdown
- Sample-wise detailed results

### 2. Error Log (Main)
**File**: `qa_error_log_TIMESTAMP.txt`

Contains for each incorrect answer:
- Question text
- Generated answer vs. standard answer
- Retrieved memory content
- Evidence from conversations
- Basic evaluation explanation
- Comprehensive error analysis (if enabled)

### 3. Category-Specific Error Logs
**File**: `qa_error_log_TIMESTAMP_CAT_<category>.txt`

Separate error logs for each question category, making it easier to analyze category-specific issues.

### 4. Memory Files
**Directory**: `memory/`

Contains character memory files:
- `<Character>_profile.txt` - Character profile information
- `<Character>_events.txt` - Extracted event memories

## Understanding Results

### Console Output

During execution, you'll see:
1. **Real-time progress** - Sample and QA processing status
2. **Real-time statistics** - Accuracy updates after each sample
3. **Final summary** - Complete results with category breakdown

Example output:
```
ENHANCED MEMORY TEST RESULTS - UNIFIED MEMAGENT
================================================================
Samples processed: 10/10
Total sessions: 50
Sessions processed: 50
Total questions: 100
Total correct: 85
Overall accuracy: 85.00%

CATEGORY-WISE ACCURACY
================================================================
Category 0                     12/ 15 (80.0%) [3 errors]
Category 1                     18/ 20 (90.0%) [2 errors]
Category 2                     25/ 30 (83.3%) [5 errors]
...
```

### Accuracy Metrics

- **Overall Accuracy**: Percentage of correctly answered questions
- **Category-wise Accuracy**: Breakdown by question type
- **Error Count**: Number of incorrect answers per category

## Troubleshooting

### API Key Issues
```bash
Error: OpenAI API key not configured
```
**Solution**: Ensure `OPENAI_API_KEY` is set in your `.env` file

### Memory Issues
```bash
Memory files already exist for characters: [...]
```
**Solution**: Use `--force-resum` to regenerate memories, or delete memory files manually

### Rate Limiting
If you encounter rate limits, reduce `--max-workers`:
```bash
python locomo_test.py --max-workers 1
```

### Data File Not Found
```bash
FileNotFoundError: data/locomo10.json
```
**Solution**: Ensure data file exists at the specified path or use `--data-file` to specify correct path

## Advanced Usage

### Full Pipeline with Custom Settings
```bash
python locomo_test.py \
  --data-file data/locomo10.json \
  --sample-use "5" \
  --chat-deployment gpt-4o \
  --memory-dir custom_memory \
  --max-workers 3 \
  --category "1,2,3" \
  --use-profile search \
  --analyze-on all
```

### Testing Specific Scenarios
```bash
# Test only on category 0 (factual questions)
python locomo_test.py --category "0"

# Quick test on first sample with detailed analysis
python locomo_test.py --sample-use "1" --analyze-on all

# Regenerate memories and test without evaluation
python locomo_test.py --force-resum --no-eval
```

## Performance Tips

1. **Parallel Processing**: Increase `--max-workers` (3-10) for faster processing on powerful machines
2. **Sample Selection**: Use `--sample-use` to test on subset during development
3. **Category Focus**: Use `--category` to focus on specific question types
4. **Skip Evaluation**: Use `--no-eval` when only updating memories

## Project Structure

```
.
├── locomo_test.py          # Main test script
├── mem_agent.py            # Memory management agent
├── response_agent.py       # Question answering agent
├── evaluate_agent.py       # Answer evaluation agent
├── llm_factory.py          # LLM client factory
├── data/
│   └── locomo10.json      # Test data
├── memory/                 # Character memory files
├── prompts/               # System prompts
└── README.md              # This file
```

## Citation

If you use this testing framework, please cite the LoCoMo benchmark paper:

```bibtex
@article{locomo2024,
  title={LoCoMo: Long-term Conversational Memory Benchmark},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

[Specify your license here]

## Support

For issues and questions:
- Check error logs for detailed failure information
- Review the comprehensive evaluation output
- Examine memory files to verify extraction quality

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style (English comments/docs)
- All tests pass
- Error logging is comprehensive


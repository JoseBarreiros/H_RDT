# Lingxuan References Cleanup

All hardcoded references to "lingxuan" have been removed from the codebase and replaced with generic paths or environment variables.

## Changes Made

### 1. Configuration Files (`*.yaml`)

**Files Updated:**
- `configs/hrdt_pretrain.yaml`
- `configs/hrdt_finetune.yaml`
- `inference/real_example/utils/hrdt.yaml`
- `inference/robotwin2_example/H-RDT/utils/hrdt.yaml`

**Changes:**
```yaml
# Before
buf_path: /ssd/lingxuan/data/buffer

# After
buf_path: /tmp/hrdt_buffer
```

### 2. Training Scripts

**Files Updated:**
- `pretrain.sh`
- `finetune.sh`

**Changes:**
```bash
# Before
export CUTLASS_PATH="/data/lingxuan/cutlass"

# After
export CUTLASS_PATH="${CUTLASS_PATH:-/usr/local/cutlass}"
```

### 3. T5 Encoder

**Files Updated:**
- `models/encoder/t5_encoder.py`
- `inference/robotwin2_example/H-RDT/models/encoder/t5_encoder.py`

**Changes:**
```python
# Before
available_models = ["/data/lingxuan/weights/t5-v1_1-xxl", "google/t5-v1_1-xxl"]

# After
available_models = ["google/t5-v1_1-xxl"]
```

### 4. Language Encoding Script

**Files Updated:**
- `datasets/pretrain/encode_lang_batch.py`

**Changes:**
```python
# Before
MODEL_PATH = os.environ.get('T5_MODEL_PATH', "/data/lingxuan/weights/t5-v1_1-xxl")

# After
MODEL_PATH = os.environ.get('T5_MODEL_PATH', "google/t5-v1_1-xxl")
```

### 5. Setup Scripts

**Files Updated:**
- `setup_robotwin2_finetune.sh`

**Changes:**
```python
# Changed replacement to use environment variable with default
content = content.replace(
    'export T5_MODEL_PATH="/data/lingxuan/weights/t5-v1_1-xxl"',
    f'export T5_MODEL_PATH="${T5_MODEL_PATH:-google/t5-v1_1-xxl}"'
)
```

## Summary

All references to user-specific paths have been replaced with:
1. **Environment variables**: Using `${VAR:-default}` syntax for flexibility
2. **Generic paths**: `/tmp/hrdt_buffer` for buffers, standard locations otherwise
3. **HuggingFace models**: Using public model names like `google/t5-v1_1-xxl`

## How to Customize Paths

### Buffer Path
Override in your shell or scripts:
```bash
# Option 1: Edit YAML directly
vim configs/hrdt_pretrain.yaml
# Change: buf_path: /tmp/hrdt_buffer
# To: buf_path: /your/custom/path

# Option 2: Set via environment variable (if supported in your code)
export HRDT_BUFFER_PATH="/your/custom/path"
```

### CUTLASS Path
```bash
export CUTLASS_PATH="/your/custom/cutlass"
./pretrain.sh
```

### T5 Model
```bash
export T5_MODEL_PATH="google/t5-v1_1-xxl"
# or for a local model
export T5_MODEL_PATH="/path/to/local/t5-model"
```

## Verification

You can verify all "lingxuan" references have been removed:

```bash
grep -r "lingxuan" --include="*.py" --include="*.sh" --include="*.yaml" .
```

Should only show references in documentation files (FIXES_APPLIED.md, this file).


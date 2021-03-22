# ONNX-ML Demo

## Train

Train models using synthetic data:

```
python3 train/train.py
```

Model files will be generated in `models` directory.

## Inference

This will load model from `models` directory and run inference on synthetic
data:

```
python3 train/infer.py
```

## Convert Models and Run Inference using ONNXRuntime


### scikit-learn models

Convert models

```
python3 converter/convert_basic.py models/scaler.pkl models/scaler.onnx
python3 converter/convert_basic.py models/clf.pkl models/clf.onnx
```

Convert RandomForestClassifier model without ZipMap operator

```
python3 converter/convert_basic.py models/clf.pkl models/clf_no_zipmap.onnx no_zipmap
```

Run inference on ONNX models (Python)

```
# Generate 10 x 3 input tensor
python3 converter/inference_basic.py models/scaler.onnx 10
python3 converter/inference_basic.py models/clf.onnx 10

# Run on a larger tensor, specify "q" to supress inference result
python3 converter/inference_basic.py models/scaler.onnx 1000000 q
python3 converter/inference_basic.py models/clf.onnx 1000000 q
python3 converter/inference_basic.py models/clf_no_zipmap.onnx 1000000 q
```

Run inference on ONNX models (C++)

```
# Build executable from source
make -C cpp infer_basic

# Generate 10 x 3 input tensor
./cpp/infer_basic models/scaler.onnx 10
./cpp/infer_basic models/clf.onnx 10

# Run on a larger tensor, specify "q" to supress inference result
./cpp/infer_basic models/scaler.onnx 1000000 q
./cpp/infer_basic models/clf.onnx 1000000 q
./cpp/infer_basic models/clf_no_zipmap.onnx 1000000 q
```

### Convert Multinomial HMM Model

Convert model

Run naive converter implemented using onnx helper functions:

```
python3 converter/convert_hmm_naive.py models/hmm.pkl models/hmm.onnx
```

or convert using sklearn-onnx custom converter:

```
python3 converter/convert_hmm_skl2onnx.py models/hmm.pkl models/hmm.onnx
```

Build custom operator as a shared library

```
make -C cpp libviterbi.so
```

Run inference on HMM ONNX model:

```
python3 converter/inference_hmm.py models/hmm.onnx 10
```

## Convert and Inference the Entire Pipeline

Combine all submodels to one ONNX model:

```
python3 converter/convert_pipeline.py models/scaler.pkl models/clf.pkl models/hmm.pkl models/pipeline.onnx
```

Run inference on ONNX model (Python)

```
python3 converter/inference_pipeline.py  models/pipeline.onnx 10
```

Run inference on ONNX model (C++)

```
./cpp/infer_basic models/pipeline.onnx 10
```

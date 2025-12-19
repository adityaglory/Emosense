import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Wrapper Class
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        return output.logits

def convert_to_onnx():
    model_path = "./artifacts/model"
    onnx_path = "./artifacts/model.onnx"
    quantized_path = "./artifacts/model_quant.onnx"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Please Run train.py!")

    for f in [onnx_path, quantized_path]:
        if os.path.exists(f): os.remove(f)

    print("🔄 Loading PyTorch Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model = ModelWrapper(base_model)
    model.eval()

    text = "Machine learning is fascinating"
    inputs = tokenizer(text, return_tensors="pt")
    
    # 1. Export to ONNX
    print("📦 Exporting to ONNX...")
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        },
        opset_version=18,
        do_constant_folding=True
    )
    
    # 2. SANITAZE METADATA
    print("🧹 Cleaning ONNX Graph Metadata...")
    onnx_model = onnx.load(onnx_path)
    del onnx_model.graph.value_info[:]
    onnx.save(onnx_model, onnx_path)
    print("✅ Graph cleaned!")

    # 3. Quantization
    print("📉 Starting Quantization (FP32 -> INT8)...")
    try:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8
        )
        print(f"✅ Ready at: {quantized_path}")
            
    except Exception as e:
        print(f"\n❌ Quantization Error: {e}")
        # Fallback
        import shutil
        shutil.copy(onnx_path, quantized_path)
        print("⚠️ Copied standard model to model_quant.onnx as fallback.")

if __name__ == "__main__":
    convert_to_onnx()
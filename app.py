import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

class InferlessPythonModel:
  def initialize(self):
      model_id = "TheBloke/vicuna-7B-v1.3-GPTQ"
      snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
      self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
      self.model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        use_safetensors=True,
        device="cuda:0",
        quantize_config=None,
        inject_fused_attention=False
      )

  def infer(self, inputs):
    prompt = inputs["prompt"]
    input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = self.model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=200)
    result = self.tokenizer.decode(output[0])
    return {"generated_result": result}

  def finalize(self,args):
    self.model = None

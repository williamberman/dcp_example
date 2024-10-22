import os
import gc
import requests
import tqdm
from PIL import Image
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

# torchrun --nproc_per_node 8 dcp_example.py

device = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(device)

def main():
    dist.init_process_group(backend='nccl', device_id=torch.device(f"cuda:{device}"))

    processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-896")
    model = FSDP(
        PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-896").train().requires_grad_(True).to(device),
        use_orig_params=True,
        device_mesh=init_device_mesh("cuda", (dist.get_world_size(),)),
        auto_wrap_policy=lambda module, recurse, nonwrapped_numel: recurse or isinstance(module, (SiglipEncoderLayer, GemmaDecoderLayer))
    )
    optimizer = AdamW(model.parameters(), lr=0.1)
    lr_scheduler = LambdaLR(optimizer, lambda step: 1)

    if device == 0:
        print(f"param check: {next(model.parameters()).abs().sum().item()}")

    dist.barrier()

    input = processor(
        text="caption es", 
        images=Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true", stream=True).raw),
        return_tensors="pt"
    ).to(device)

    for _ in tqdm.tqdm(range(10)):
        output = model(**input)
        loss = output.logits[0, 0, :].sum()
        loss.backward()
        model.clip_grad_norm_(1.0)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    dist.barrier()

    if device == 0:
        print(f"after training param check: {next(model.parameters()).abs().sum().item()}")

    dcp.save(dict(state=Stateful_(model, optimizer, lr_scheduler)), checkpoint_id='./test_ckpt/')

    del model, optimizer, lr_scheduler

    gc.collect()
    torch.cuda.empty_cache()

    model = FSDP(
        PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-896").train().requires_grad_(True).to(device),
        use_orig_params=True,
        device_mesh=init_device_mesh("cuda", (dist.get_world_size(),)),
        auto_wrap_policy=lambda module, recurse, nonwrapped_numel: recurse or isinstance(module, (SiglipEncoderLayer, GemmaDecoderLayer))
    )
    optimizer = AdamW(model.parameters(), lr=0.1)
    lr_scheduler = LambdaLR(optimizer, lambda step: 1)

    if device == 0:
        print(f"recreate model param check: {next(model.parameters()).abs().sum().item()}")

    dcp.load(dict(state=Stateful_(model, optimizer, lr_scheduler)), checkpoint_id='./test_ckpt/')

    if device == 0:
        print(f"loaded ckpt param check: {next(model.parameters()).abs().sum().item()}")

    dist.destroy_process_group()

from torch.distributed.checkpoint.stateful import Stateful
class Stateful_(Stateful):
    def __init__(self, model, optimizer, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        from torch.distributed.checkpoint.state_dict import get_state_dict
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optimizer": optimizer_state_dict, "lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        from torch.distributed.checkpoint.state_dict import set_state_dict
        set_state_dict(self.model, self.optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

if __name__ == "__main__":
    main()
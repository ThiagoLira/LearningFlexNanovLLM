import torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel
)


class ToyMLPSharded(nn.Module):
    def __init__(self, hidden_size: int, ff_size: int ) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, ff_size)
        self.down_proj = nn.Linear(ff_size, hidden_size)

    def forward(self, x):
        x = torch.relu(self.up_proj(x))
        return self.down_proj(x)


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    # 1D tensor-parallel mesh over local GPUs
    tp_mesh = init_device_mesh("cuda", (world,))

    model = ToyMLPSharded(500,1000).cuda()
    # Tell TP how to shard submodules by FQN
    tp_plan = {"up_proj": ColwiseParallel(), "down_proj": RowwiseParallel()}
    model = parallelize_module(model, tp_mesh, tp_plan)

    # Use as usual; inputs can be regular tensors on the local device
    x = torch.randn(8, 500, device="cuda")
    y = model(x)
    result = y
    # result is the same on both GPUs, the all-gather already happened
    print(result)
if __name__ == "__main__":
    import os
    main()

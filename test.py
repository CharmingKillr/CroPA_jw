import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np

npy = np.load('/var/lib/kubelet/jw/projects/CroPA_jw/output/open_flamingo_shots_0/cropa/num_10_unknown/frac_0.05/run_1/900/262162_.npy')
print(npy.shape)
# inputs_embeds = torch.rand(1, 55, 4096,requires_grad=True)
# # mask = torch.ones_like(inputs_embeds)
# # context_token_len = 30

# # mask[:,:context_token_len] = 0
# # print(mask)

# # mask[:,-2:] = 0
# inputs_embeds_original = torch.rand(1, 55, 4096)
# text_perturb = torch.zeros_like(inputs_embeds_original,requires_grad=True)
# # 假设 inputs_embeds = inputs_embeds_original + text_perturb

# inputs_embeds = inputs_embeds_original + text_perturb

# inputs_embeds.retain_grad()
# # 假设 loss 是 inputs_embeds 的简单函数，比如平方和
# loss = torch.sum(inputs_embeds**2)

# # 反向传播
# loss.backward()

# # 验证梯度是否相等
# print(torch.allclose(inputs_embeds.grad, text_perturb.grad))  # 应输出 True

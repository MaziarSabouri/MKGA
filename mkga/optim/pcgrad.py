import torch
import random


class PCGrad:
    """Projecting Conflicting Gradients wrapper for PyTorch optimizers."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()

    def pc_backward(self, losses):
        assert len(losses) > 1, "PCGrad requires at least 2 losses."
        num_tasks = len(losses)
        task_grads = []
        for loss in losses:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            grad_vector = []
            for p in self.optimizer.param_groups[0]['params']:
                if p.grad is not None:
                    grad_vector.append(p.grad.clone().view(-1))
                else:
                    grad_vector.append(torch.zeros_like(p).view(-1))
            task_grads.append(torch.cat(grad_vector))
        self.optimizer.zero_grad(set_to_none=True)

        task_order = list(range(num_tasks))
        random.shuffle(task_order)
        projected_grads = []
        for i in task_order:
            g_i = task_grads[i].clone()
            for g_j in projected_grads:
                dot_prod = torch.dot(g_i, g_j)
                if dot_prod < 0:
                    norm_j_sq = torch.dot(g_j, g_j)
                    g_i = g_i - (dot_prod / (norm_j_sq + 1e-8)) * g_j
            projected_grads.append(g_i)

        total_grad = torch.sum(torch.stack(projected_grads), dim=0)
        offset = 0
        for p in self.optimizer.param_groups[0]['params']:
            if p.requires_grad:
                numel = p.numel()
                p.grad = total_grad[offset: offset + numel].view_as(p).clone()
                offset += numel

import torch
import torch.nn.functional as F
from trl import DPOTrainer

class HypoDPOTrainer(DPOTrainer):
    def __init__(self, *args, reference_free: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_free = reference_free

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """HyPO loss (Hard HyPO version: max(ref_logratios, 0))"""
        pi_logratios = policy_chosen_logps - policy_rejected_logps

        if self.reference_free:
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        # Hard HyPO: 对 ref_logratios 进行裁剪（论文 Eq. 11）
        ref_logratios = torch.maximum(ref_logratios, torch.tensor(0.0, device=ref_logratios.device))

        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits)

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss.mean(), chosen_rewards, rejected_rewards
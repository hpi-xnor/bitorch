# Code is modified from MEAL (https://arxiv.org/abs/1812.02425) and Label Refinery (https://arxiv.org/abs/1805.02641).

import torch
from torch.nn import functional as F
from torch.nn.modules import loss


class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for a student and teacher model."""

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        """
        Calculate the KL-Divergence loss.

        Args:
            student_out: NxC tensor (must be the output of the student network before softmax function)
            teacher_out: NxC tensor (each row must be a probability score, adding up to one)

        Returns:
            the loss score
        """
        # check that teacher does not require gradients
        if teacher_out.requires_grad:
            raise ValueError("real network output should not require gradients.")

        student_log_prob = F.log_softmax(student_out, dim=1)
        teacher_soft_output = F.softmax(teacher_out, dim=1)
        del student_out, teacher_out

        # Loss is -dot(student_log_prob, teacher_out). Reshape tensors for batch matrix multiplication
        teacher_soft_output = teacher_soft_output.unsqueeze(1)
        student_log_prob = student_log_prob.unsqueeze(2)

        # Compute the loss, and average for the batch.
        cross_entropy_loss = -torch.bmm(teacher_soft_output, student_log_prob)

        return cross_entropy_loss.mean()

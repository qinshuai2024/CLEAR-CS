import sys
import os
sys.path += ['../']
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,AutoConfig,AutoModel


class QANLIReranker(nn.Module):
    """Use extractive QA model as discriminator + entailment head"""
    def __init__(
        self,
        model_name: str = "deepset/roberta-base-squad2",
        use_qa_logits_score: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.use_qa_logits_score = use_qa_logits_score

        hidden_size = self.qa_model.config.hidden_size
        # Output a scalar representing entailment probability
        # self.entail_head = nn.Linear(hidden_size, 1)
        self.entail_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Input:
            input_ids: (N, L) or (N, M, L)
            attention_mask: same as input_ids
        Output:
            relevance_scores: (N,) or (N, M)
            entail_probs: (N,) or (N, M)
        """
        original_shape = input_ids.shape
        is_batched = len(original_shape) == 3
        if is_batched:
            N, M, L = input_ids.shape
            # Use clone to avoid in-place operations
            input_ids = input_ids.clone().view(N * M, L)
            attention_mask = attention_mask.clone().view(N * M, L)

        # Forward to QA model
        outputs = self.qa_model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True, return_dict=True)
        start_logits = outputs.start_logits  # (B, L)
        end_logits = outputs.end_logits      # (B, L)

        # relevance score
        start_max = torch.max(start_logits, dim=-1).values
        end_max = torch.max(end_logits, dim=-1).values
        relevance_scores = start_max + end_max  # (B,)

        # entailment probability (sigmoid)
        cls_repr = outputs.hidden_states[-1][:, 0, :]   # [CLS]
        entail_logits = self.entail_head(cls_repr)      # (B, 1)
        entail_probs = torch.sigmoid(entail_logits).squeeze(-1)  # (B,)

        if is_batched:
            relevance_scores = relevance_scores.view(N, M)
            entail_probs = entail_probs.view(N, M)

        return relevance_scores, entail_probs



# class QANLI2WayReranker(nn.Module):
#     """Binary classification reranking model based on extractive QA encoder + NLI distillation head
#     - Input: (N, L) or (N, M, L)
#     - Output: class_logits: (N, 2) or (N, M, 2); entail_probs: (N,) or (N, M)
#     """
#     def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
#         super().__init__()
#         self.model_name = model_name
#         self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#         hidden_size = self.qa_model.config.hidden_size

#         # # 4-class relevance head & NLI distillation head
#         # self.classifier = nn.Linear(hidden_size, 2)
#         # self.entail_head = nn.Linear(hidden_size, 1)

#         # Use GELU activation function
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, 2)
#         )
#         self.entail_head = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, 1)
#         )

#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
#         original_shape = input_ids.shape
#         is_batched = len(original_shape) == 3

#         if is_batched:
#             N, M, L = original_shape
#             input_ids = input_ids.clone().view(N * M, L)
#             attention_mask = attention_mask.clone().view(N * M, L)

#         # Only use RoBERTa encoder's hidden states
#         base_outputs = self.qa_model.roberta(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )
#         hidden = base_outputs.last_hidden_state  # (B, L, H)
#         cls_repr = hidden[:, 0, :]              # (B, H)

#         class_logits = self.classifier(cls_repr)              # (B, 2)
#         entail_probs = torch.sigmoid(self.entail_head(cls_repr)).squeeze(-1)  # (B,)

#         if is_batched:
#             class_logits = class_logits.view(N, M, 2)
#             entail_probs = entail_probs.view(N, M)

#         return class_logits, entail_probs


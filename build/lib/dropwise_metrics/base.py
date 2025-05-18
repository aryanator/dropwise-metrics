import torch
import torch.nn.functional as F
from torchmetrics import Metric

from dropwise_metrics.tasks import get_task_handler  # Your original handler router


class DropwiseBaseMetric(Metric):
    def __init__(
        self,
        model,
        tokenizer,
        task_type: str = "sequence-classification",
        num_passes: int = 20,
        use_cuda: bool = True
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.num_passes = num_passes
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        self.model.to(self._device)
        self.model.eval()
        self._enable_mc_dropout()

        self.task_handler = get_task_handler(task_type)
        self.add_state("texts", default=[], dist_reduce_fx="cat")

    def _enable_mc_dropout(self):
        """Activates dropout layers at inference time (MC Dropout logic)."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def update(self, texts):
        """Stores texts to be processed in compute()."""
        if isinstance(texts, str):
            texts = [texts]
        self.texts.extend(texts)

    def run_mc_inference(self):
        """Tokenizes and runs MC Dropout inference N times."""
        inputs = self.tokenizer(self.texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        all_logits = []

        with torch.no_grad():
            for _ in range(self.num_passes):
                outputs = self.model(**inputs)

                if self.task_type == "question-answering":
                    combined = torch.stack([outputs.start_logits, outputs.end_logits], dim=-1)
                    all_logits.append(combined)
                else:
                    all_logits.append(outputs.logits)

        stacked_logits = torch.stack(all_logits)  # [num_passes, batch, ...]
        return stacked_logits, inputs

    def decode_task_outputs(self, stacked_logits, inputs):
        """Calls your Dropwise task handler to produce per-sample results."""
        mean_logits = stacked_logits.mean(dim=0)
        std_logits = stacked_logits.std(dim=0)

        if "classification" in self.task_type:
            all_probs = torch.stack([F.softmax(logits, dim=-1) for logits in stacked_logits])
            probs = all_probs.mean(dim=0)
        else:
            probs = mean_logits
            all_probs = None

        results = self.task_handler(
            inputs=inputs,
            probs=probs,
            mean_logits=mean_logits,
            std_logits=std_logits,
            text=self.texts,
            tokenizer=self.tokenizer,
            custom_metrics={},  # extend later
            verbose=False
        )
        return results

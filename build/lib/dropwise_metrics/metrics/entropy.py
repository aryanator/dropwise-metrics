from dropwise_metrics.base import DropwiseBaseMetric

class PredictiveEntropyMetric(DropwiseBaseMetric):
    def compute(self):
        """
        Runs MC dropout inference and returns per-sample outputs
        including entropy, predicted class, confidence, etc.
        """
        stacked_logits, inputs = self.run_mc_inference()
        return self.decode_task_outputs(stacked_logits, inputs)

from utils.logger import get_logger

logger = get_logger("PromptParser")

class PromptParser:
    def __init__(self):
        # Keyword mappings for heuristic analysis
        self.fast_keywords = ["fast", "quick", "low latency", "speed", "real-time"]
        self.efficient_keywords = ["efficient", "lightweight", "small", "low memory", "mobile", "edge"]
        self.accurate_keywords = ["accurate", "best", "high performance", "perfect", "state-of-the-art", "highly accurate", "high accuracy"]
        self.deep_keywords = ["deep", "complex", "powerful"]
        self.robust_keywords = ["robust", "avoid errors", "minimize classification errors", "generalization", "reliability"]

    def parse(self, prompt_text, default_weights=None):
        logger.info(f"Parsing prompt: '{prompt_text}'")

        # Base weights for fitness function F(A) = α*acc - β*params - γ*latency - δ*memory
        # plus architecture multipliers based on prompt
        weights = default_weights or {
            "alpha": 1.0,  # Accuracy weight
            "beta": 0.1,   # Params penalty
            "gamma": 0.1,  # Latency penalty
            "delta": 0.1,   # Memory penalty
            "epoch_multiplier": 1,
            "dropout_rate": 0.2,
            "max_params_multiplier": 1.0
        }

        prompt_lower = prompt_text.lower()

        # Apply heuristics based on keywords
        is_fast = any(kw in prompt_lower for kw in self.fast_keywords)
        is_efficient = any(kw in prompt_lower for kw in self.efficient_keywords)
        is_accurate = any(kw in prompt_lower for kw in self.accurate_keywords)

        is_robust = any(kw in prompt_lower for kw in self.robust_keywords)

        if is_fast:
            logger.info("Detected 'fast' requirement. Increasing latency penalty (gamma).")
            weights["gamma"] += 0.5
            weights["alpha"] -= 0.2  # Trade-off accuracy for speed

        if is_efficient:
            logger.info("Detected 'efficient' requirement. Increasing memory/params penalty (beta, delta).")
            weights["beta"] += 0.4
            weights["delta"] += 0.4

        if is_accurate:
            logger.info("Detected 'accurate' requirement. Prioritizing performance over cost.")
            weights["alpha"] += 0.8
            weights["beta"] = max(0.01, weights["beta"] - 0.08)
            weights["gamma"] = max(0.01, weights["gamma"] - 0.08)
            weights["delta"] = max(0.01, weights["delta"] - 0.08)
            weights["epoch_multiplier"] = 3  # Train 3x longer
            weights["max_params_multiplier"] = 2.0  # Allow double the parameters (deeper networks)

        if is_robust:
            logger.info("Detected 'robust' requirement. Increasing dropout for generalization.")
            weights["dropout_rate"] = 0.4
            weights["epoch_multiplier"] = max(weights["epoch_multiplier"], 2)

        # Normalize weights to prevent extreme values (optional, but good for stability)
        # We ensure alpha is always positive
        weights["alpha"] = max(0.1, weights["alpha"])

        logger.info(f"Final fitness weights derived from prompt: {weights}")
        return weights

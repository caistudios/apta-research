"""Entry point for running inference of cascaded models

1. Define the model chains
2. Define the system prompts used for each
3. Get the dataset
4. Run inference
5. Save the responses
6. Evaluate the responses

"""

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from apta_research.cascade.base import RESULTS_DIR
from apta_research.common.llm import BaseLLM, BaseLLMConfig


class CascadeConfig(BaseModel):
    model_configs: list[BaseLLMConfig]
    debug: bool = Field(default=True, exclude=True)
    save_dir: Path = Field(default=RESULTS_DIR, exclude=True)

    @model_validator(mode="after")
    def set_unique_save_dir(self) -> "CascadeConfig":
        if self.save_dir == RESULTS_DIR:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = self.save_dir / "other" / timestamp

        return self

    def save(self):
        """Save the config to a file"""

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save the config to a yaml file
        config_file = self.save_dir / "config.yaml"
        config = self.model_dump(mode="json")

        # dump the config to a yaml file
        with config_file.open("w") as f:
            yaml.dump(config, f)

        if self.debug:
            print(f"Saved config to {config_file}")

        return self.save_dir


class Cascade:
    def __init__(self, config: CascadeConfig):
        self.models = [
            BaseLLM(config=model_config) for model_config in config.model_configs
        ]

    def run_inference(self):
        """Go through each model in the chain and run inference

        Make sure to pass the output of the previous model to the next model
        """
        pass

    def save_responses(self):
        pass

    def evaluate_responses(self):
        pass

"""Configuration classes and enums for dataset and model settings."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DatasetType(Enum):
    """Enumeration of supported dataset types."""
    MATHVISTA = "mathvista"
    MATHVERSE = "mathverse"
    MATHVISION = "mathvision"
    SFTSEED = "sftseed"
    HALLUSIONBENCH = "hallusionbench"
    EMMA_MATH = "emma-math"
    EMMA_CHEM = "emma-chem"
    EMMA_CODE = "emma-code"
    EMMA_PHYSICS = "emma-physics"
    MMMU_PRO_10 = "mmmu-pro-10"
    MMMU_PRO_4 = "mmmu-pro-4"
    MMMU_PRO_VISION = "mmmu-pro-vision"


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    split: str
    image_field: str
    response_field: str
    instruction_field: Optional[str] = None
    subset: Optional[str] = None
    choices_field: Optional[str] = None
    options_field: Optional[str] = None
    source_field: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str
    processor_name: str
    max_new_tokens: int = 5000
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1
    repetition_penalty: float = 1.0


def get_dataset_config(dataset_type: DatasetType) -> DatasetConfig:
    """Get dataset configuration for a given dataset type."""
    configs = {
        DatasetType.MATHVISTA: DatasetConfig(
            name="AI4Math/MathVista",
            split="testmini",
            image_field="decoded_image",
            instruction_field="query",
            response_field="answer",
            choices_field="choices"
        ),
        DatasetType.MATHVERSE: DatasetConfig(
            name="AI4Math/MathVerse",
            subset="testmini",
            split="testmini",
            image_field="image",
            instruction_field="query_cot",
            response_field="answer"
        ),
        DatasetType.MATHVISION: DatasetConfig(
            name="MathLLMs/MathVision",
            split="testmini",
            image_field="decoded_image",
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.SFTSEED: DatasetConfig(
            name="ydeng9/sft_seed",
            split="train",
            image_field="decoded_image",
            instruction_field="problem",
            response_field="answer",
            source_field="source"
        ),
        DatasetType.HALLUSIONBENCH: DatasetConfig(
            name="lmms-lab/HallusionBench",
            split="image",
            image_field="image",
            instruction_field="question",
            response_field="gt_answer"
        ),
        DatasetType.EMMA_MATH: DatasetConfig(
            name="luckychao/EMMA",
            subset="Math",
            split="test",
            image_field="image_1",
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.EMMA_CHEM: DatasetConfig(
            name="luckychao/EMMA",
            subset="Chemistry",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.EMMA_CODE: DatasetConfig(
            name="luckychao/EMMA",
            subset="Coding",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.EMMA_PHYSICS: DatasetConfig(
            name="luckychao/EMMA",
            subset="Physics",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.MMMU_PRO_10: DatasetConfig(
            name="MMMU/MMMU_Pro",
            subset="standard (10 options)",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5","image_6","image_7"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.MMMU_PRO_4: DatasetConfig(
            name="MMMU/MMMU_Pro",
            subset="standard (4 options)",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5","image_6","image_7"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.MMMU_PRO_VISION: DatasetConfig(
            name="MMMU/MMMU_Pro",
            subset="vision",
            split="test",
            image_field="image",
            response_field="answer",
            options_field="options"
        ),
    }
    return configs[dataset_type]


"""Prompt templates and functions for card generation pipeline."""

from __future__ import annotations

from packages.generator.prompts.enhancement import (
    cloze_conversion_system,
    cloze_conversion_user,
    enhancement_system,
    enhancement_user,
    mnemonic_generation_system,
    mnemonic_generation_user,
    reverse_generation_system,
    reverse_generation_user,
    split_suggestion_system,
    split_suggestion_user,
)
from packages.generator.prompts.generation import (
    card_generation_system,
    card_generation_user,
    card_splitting_system,
    card_splitting_user,
    context_enrichment_system,
    context_enrichment_user,
)
from packages.generator.prompts.templates import (
    PromptTemplate,
    create_simple_template,
    merge_templates,
    parse_template_string,
    validate_template,
)
from packages.generator.prompts.validation import (
    memorization_assessment_system,
    memorization_assessment_user,
    post_validation_system,
    post_validation_user,
    pre_validation_system,
    pre_validation_user,
)

__all__ = [
    # templates
    "PromptTemplate",
    # generation
    "card_generation_system",
    "card_generation_user",
    "card_splitting_system",
    "card_splitting_user",
    "cloze_conversion_system",
    "cloze_conversion_user",
    "context_enrichment_system",
    "context_enrichment_user",
    "create_simple_template",
    # enhancement
    "enhancement_system",
    "enhancement_user",
    "memorization_assessment_system",
    "memorization_assessment_user",
    "merge_templates",
    "mnemonic_generation_system",
    "mnemonic_generation_user",
    "parse_template_string",
    "post_validation_system",
    "post_validation_user",
    # validation
    "pre_validation_system",
    "pre_validation_user",
    "reverse_generation_system",
    "reverse_generation_user",
    "split_suggestion_system",
    "split_suggestion_user",
    "validate_template",
]

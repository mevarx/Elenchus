"""Data model for extraction results.

FLAT FIELDS ONLY. Measured on gemma3:4b: a nested-array schema
(`amounts: [{label, value}]`) returned `[]` in 3/3 runs while still
emitting a confident total. Flat, explicitly-named fields do not have
this failure mode. This is a build landmine — do not "improve" this
schema into nested structures.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class LineItem:
    description: str
    amount_inr: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Extraction:
    """What the model claims a document says."""
    line_items: list[LineItem] = field(default_factory=list)
    stated_total_inr: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Extraction":
        items = [LineItem(description=str(i.get("description", "")),
                          amount_inr=float(i.get("amount_inr", 0.0)))
                 for i in d.get("line_items", [])]
        total = d.get("stated_total_inr")
        return cls(line_items=items,
                   stated_total_inr=None if total is None else float(total))

    def to_dict(self) -> dict:
        return {"line_items": [i.to_dict() for i in self.line_items],
                "stated_total_inr": self.stated_total_inr}


# The JSON schema handed to Ollama (`format=`). Kept flat on purpose.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount_inr": {"type": "number"},
                },
                "required": ["description", "amount_inr"],
            },
        },
        "stated_total_inr": {"type": "number"},
    },
    "required": ["line_items", "stated_total_inr"],
}

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    persist_directory: str | None = None
    anonymized_telemetry: bool = False

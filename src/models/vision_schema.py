from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: List[int]
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'attributes': self.attributes,
        }


@dataclass
class VisionEvent:
    timestamp: str
    sensor_id: str
    event_type: str
    detections: List[Detection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'event_type': self.event_type,
            'detections': [d.to_dict() for d in self.detections],
        }

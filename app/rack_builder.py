from dataclasses import dataclass
from uuid import uuid4
from app.app_meta import (
    EffectSpec,
    EFFECT_REGISTRY
)

@dataclass
class RackItem:
    id: str
    name: str
    effect: object
    spec: EffectSpec
    
def create_rack_item(effect_name, fs):
    spec = EFFECT_REGISTRY[effect_name]
    effect = spec.factory(fs)
    effect.enabled = True
    return RackItem(
        id=str(uuid4()),
        name=effect_name,
        effect=effect,
        spec=spec,
    )


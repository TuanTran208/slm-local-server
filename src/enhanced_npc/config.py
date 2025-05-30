import os.path

ocean_traits = {
    'Openness': [0 - 100],  # Curiosity, creativity, abstract thinking
    'Conscientiousness': [0 - 100],  # Organization, responsibility, planning
    'Extraversion': [0 - 100],  # Social energy, assertiveness
    'Agreeableness': [0 - 100],  # Empathy, cooperation, compassion
    'Neuroticism': [0 - 100]  # Emotional stability/volatility
}
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml
from yaml import Loader


class OCEAN(BaseModel):
    openness: int
    conscientiousness: int
    extraversion: int
    agreeableness: int
    neuroticism: int


@dataclass
class CharacterConfig:
    templates: Dict[str, Any]
    character_psychology: Dict[str, Any]

    @classmethod
    def load(cls, path: str = "./src/enhanced_npc/character_info.yml") -> 'CharacterConfig':
        print(os.path.abspath("./src/enhanced_npc/character_info.yml"))
        with open(os.path.abspath(path), 'r') as f:
            data = yaml.load(f, Loader=Loader)  # Use full loader for complex types
            return cls(
                templates=data.get('templates', {}),
                character_psychology=data.get('character_psychology', {})
            )


maslow_needs = {
    'Physiological': {
        'current_state': [0 - 100],
        'priorities': ['food', 'sleep', 'shelter']
    },
    'Safety': {
        'current_state': [0 - 100],
        'priorities': ['security', 'stability', 'resources']
    },
    'Belonging': {
        'current_state': [0 - 100],
        'priorities': ['relationships', 'community', 'intimacy']
    },
    'Esteem': {
        'current_state': [0 - 100],
        'priorities': ['recognition', 'achievement', 'respect']
    },
    'Self_Actualization': {
        'current_state': [0 - 100],
        'priorities': ['growth', 'purpose', 'potential']
    }
}

plutchik_emotions = {
    'primary': {
        'joy': [0 - 100],
        'trust': [0 - 100],
        'fear': [0 - 100],
        'surprise': [0 - 100],
        'sadness': [0 - 100],
        'disgust': [0 - 100],
        'anger': [0 - 100],
        'anticipation': [0 - 100]
    },
    'complex': {
        'love': ['joy', 'trust'],
        'submission': ['trust', 'fear'],
        'awe': ['fear', 'surprise'],
        'disappointment': ['surprise', 'sadness'],
        'remorse': ['sadness', 'disgust'],
        'contempt': ['disgust', 'anger'],
        'aggressiveness': ['anger', 'anticipation'],
        'optimism': ['anticipation', 'joy']
    }
}

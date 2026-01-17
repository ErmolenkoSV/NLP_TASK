"""
Базовый класс для NER моделей
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseNER(ABC):
    """Базовый абстрактный класс для всех NER моделей"""
    
    def __init__(self, model_name: str = "base"):
        """
        Args:
            model_name: название модели
        """
        self.model_name = model_name
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлечь именованные сущности из текста
        
        Args:
            text: входной текст
            
        Returns:
            List[Dict]: список сущностей в формате:
                [
                    {
                        "text": "название сущности",
                        "type": "тип сущности (PERSON, ORG, LOC, ...)",
                        "start": позиция начала,
                        "end": позиция конца
                    },
                    ...
                ]
        """
        pass
    
    def extract_entities_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Извлечь сущности из списка текстов
        
        Args:
            texts: список текстов
            
        Returns:
            List[List[Dict]]: список списков сущностей
        """
        return [self.extract_entities(text) for text in texts]
    
    def get_entity_types(self) -> List[str]:
        """
        Получить список поддерживаемых типов сущностей
        
        Returns:
            List[str]: список типов
        """
        return ["PERSON", "ORG", "LOC", "DATE", "MONEY", "PRODUCT", "EVENT"]
    
    def __str__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
    
    def __repr__(self):
        return self.__str__()


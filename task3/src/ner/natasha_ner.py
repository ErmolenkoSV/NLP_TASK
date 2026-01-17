"""
NER с использованием библиотеки Natasha (для русского языка)
"""

from typing import List, Dict, Any
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)
from .base_ner import BaseNER


class NatashaNER(BaseNER):
    """NER с использованием библиотеки Natasha (специально для русского языка)"""
    
    def __init__(self):
        """Инициализация Natasha NER"""
        super().__init__(model_name="natasha")
        
        print("Загрузка Natasha моделей...")
        
        # Инициализация компонентов Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)
        
        print("Модели Natasha загружены")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлечь именованные сущности из текста
        
        Args:
            text: входной текст
            
        Returns:
            List[Dict]: список сущностей
        """
        # Создаем документ
        doc = Doc(text)
        
        # Сегментация
        doc.segment(self.segmenter)
        
        # NER
        doc.tag_ner(self.ner_tagger)
        
        # Извлекаем сущности
        entities = []
        for span in doc.spans:
            entities.append({
                "text": span.text,
                "type": self._map_entity_type(span.type),
                "start": span.start,
                "end": span.stop
            })
        
        return entities
    
    def _map_entity_type(self, natasha_label: str) -> str:
        """
        Маппинг типов сущностей Natasha на стандартные типы
        
        Args:
            natasha_label: тип сущности Natasha
            
        Returns:
            str: стандартный тип
        """
        mapping = {
            "PER": "PERSON",
            "ORG": "ORG",
            "LOC": "LOC",
            "DATE": "DATE",
            "MONEY": "MONEY"
        }
        
        return mapping.get(natasha_label, natasha_label)
    
    def get_entity_types(self) -> List[str]:
        """Получить поддерживаемые типы сущностей"""
        return ["PERSON", "ORG", "LOC"]


# Пример использования
if __name__ == "__main__":
    # Создаем NER модель
    ner = NatashaNER()
    
    # Тестовый текст
    text = "Владимир Путин посетил Москву 1 января 2023 года и встретился с представителями компании Яндекс."
    
    # Извлекаем сущности
    entities = ner.extract_entities(text)
    
    print("Найденные сущности:")
    for entity in entities:
        print(f"  - {entity['text']:<30} ({entity['type']})")


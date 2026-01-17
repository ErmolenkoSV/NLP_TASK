"""
NER с использованием spaCy
"""

from typing import List, Dict, Any
import spacy
from .base_ner import BaseNER


class SpacyNER(BaseNER):
    """NER с использованием предобученных моделей spaCy"""
    
    def __init__(self, model_name: str = "ru_core_news_sm", language: str = "ru"):
        """
        Args:
            model_name: название spaCy модели
                - ru_core_news_sm/md/lg (русский)
                - en_core_web_sm/md/lg (английский)
            language: язык текста
        """
        super().__init__(model_name=model_name)
        self.language = language
        
        print(f"Загрузка spaCy модели {model_name}...")
        
        try:
            self.nlp = spacy.load(model_name)
            print(f"Модель загружена: {model_name}")
        except OSError:
            print(f"Модель {model_name} не найдена. Установите её:")
            print(f"python -m spacy download {model_name}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлечь именованные сущности из текста
        
        Args:
            text: входной текст
            
        Returns:
            List[Dict]: список сущностей
        """
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": self._map_entity_type(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def _map_entity_type(self, spacy_label: str) -> str:
        """
        Маппинг типов сущностей spaCy на стандартные типы
        
        Args:
            spacy_label: тип сущности spaCy
            
        Returns:
            str: стандартный тип
        """
        # Маппинг для русских меток
        mapping_ru = {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "ORG": "ORG",
            "LOC": "LOC",
            "GPE": "LOC",
            "DATE": "DATE",
            "TIME": "DATE",
            "MONEY": "MONEY",
            "PERCENT": "MONEY",
            "PRODUCT": "PRODUCT",
            "EVENT": "EVENT"
        }
        
        # Маппинг для английских меток
        mapping_en = {
            "PERSON": "PERSON",
            "ORG": "ORG",
            "GPE": "LOC",
            "LOC": "LOC",
            "DATE": "DATE",
            "TIME": "DATE",
            "MONEY": "MONEY",
            "PERCENT": "MONEY",
            "PRODUCT": "PRODUCT",
            "EVENT": "EVENT",
            "FAC": "LOC",
            "NORP": "ORG"
        }
        
        mapping = mapping_ru if self.language == "ru" else mapping_en
        return mapping.get(spacy_label, "OTHER")
    
    def get_entity_types(self) -> List[str]:
        """Получить поддерживаемые типы сущностей"""
        return list(self.nlp.pipe_labels.get("ner", []))


# Пример использования
if __name__ == "__main__":
    # Создаем NER модель
    ner = SpacyNER(model_name="ru_core_news_sm", language="ru")
    
    # Тестовый текст
    text = "Владимир Путин посетил Москву 1 января 2023 года и встретился с представителями компании Яндекс."
    
    # Извлекаем сущности
    entities = ner.extract_entities(text)
    
    print("Найденные сущности:")
    for entity in entities:
        print(f"  - {entity['text']:<30} ({entity['type']})")


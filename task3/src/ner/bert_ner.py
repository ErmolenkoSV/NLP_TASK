"""
NER с использованием BERT-like моделей
"""

from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from .base_ner import BaseNER


class BERTNER(BaseNER):
    """NER с использованием BERT-подобных моделей"""
    
    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        device: str = "auto",
        aggregation_strategy: str = "simple"
    ):
        """
        Args:
            model_name: название модели с Hugging Face
                - dslim/bert-base-NER (английский)
                - DeepPavlov/rubert-base-cased-conversational (русский)
            device: устройство ("cpu", "cuda", "auto")
            aggregation_strategy: стратегия агрегации токенов
        """
        super().__init__(model_name=model_name)
        
        self.device = device
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            self.device = 0
        else:
            self.device = -1
        
        print(f"Загрузка BERT модели {model_name}...")
        
        # Создаем pipeline для NER
        self.nlp = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            aggregation_strategy=aggregation_strategy
        )
        
        print(f"Модель загружена на {'cuda' if self.device >= 0 else 'cpu'}")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлечь именованные сущности из текста
        
        Args:
            text: входной текст
            
        Returns:
            List[Dict]: список сущностей
        """
        # Ограничиваем длину текста (BERT имеет ограничение 512 токенов)
        if len(text) > 2000:
            text = text[:2000]
        
        # Извлекаем сущности
        results = self.nlp(text)
        
        entities = []
        for result in results:
            entities.append({
                "text": result["word"],
                "type": self._map_entity_type(result["entity_group"]),
                "start": result["start"],
                "end": result["end"],
                "score": result.get("score", 1.0)
            })
        
        return entities
    
    def _map_entity_type(self, bert_label: str) -> str:
        """
        Маппинг типов сущностей BERT на стандартные типы
        
        Args:
            bert_label: тип сущности BERT
            
        Returns:
            str: стандартный тип
        """
        # Убираем префиксы B-, I-
        label = bert_label.replace("B-", "").replace("I-", "")
        
        mapping = {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "ORG": "ORG",
            "LOC": "LOC",
            "GPE": "LOC",
            "MISC": "OTHER",
            "DATE": "DATE",
            "TIME": "DATE",
            "MONEY": "MONEY",
            "PERCENT": "MONEY"
        }
        
        return mapping.get(label, label)


# Пример использования
if __name__ == "__main__":
    # Создаем NER модель
    ner = BERTNER(
        model_name="dslim/bert-base-NER",
        device="cpu"
    )
    
    # Тестовый текст
    text = "My name is Wolfgang and I live in Berlin. I work at Google."
    
    # Извлекаем сущности
    entities = ner.extract_entities(text)
    
    print("Найденные сущности:")
    for entity in entities:
        print(f"  - {entity['text']:<30} ({entity['type']}) [score: {entity.get('score', 0):.2f}]")


"""
NER с использованием локальных LLM моделей
"""

import json
import re
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm

from .base_ner import BaseNER
from ..utils.prompts import get_ner_prompt


class LLMNER(BaseNER):
    """NER с использованием локальных LLM (<= 7B параметров)"""
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_length: int = 2048,
        temperature: float = 0.1,
        language: str = "ru"
    ):
        """
        Args:
            model_name: название модели с Hugging Face
            device: устройство ("cpu", "cuda", "auto")
            load_in_8bit: загрузить в 8-bit (экономия памяти)
            load_in_4bit: загрузить в 4-bit (экономия памяти)
            max_length: максимальная длина генерации
            temperature: температура для генерации
            language: язык текста ("ru" или "en")
        """
        super().__init__(model_name=model_name)
        
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.language = language
        
        print(f"Загрузка модели {model_name}...")
        
        # Определение устройства
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Настройки квантизации
        kwargs = {}
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        
        # Загрузка токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device if device != "cpu" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **kwargs
        )
        
        # Если нет квантизации и устройство CPU, перемещаем модель
        if not load_in_8bit and not load_in_4bit and device == "cpu":
            self.model = self.model.to(device)
        
        print(f"Модель загружена на {self.device}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Генерация ответа от LLM
        
        Args:
            prompt: промпт
            
        Returns:
            str: ответ модели
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Удаляем промпт из ответа
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Парсинг JSON из ответа модели
        
        Args:
            response: ответ модели
            
        Returns:
            Dict: распарсенный JSON
        """
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Если JSON не найден, возвращаем пустой результат
                return {"entities": []}
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            print(f"Ответ модели: {response[:200]}...")
            return {"entities": []}
    
    def extract_entities(self, text: str, use_few_shot: bool = False) -> List[Dict[str, Any]]:
        """
        Извлечь именованные сущности из текста
        
        Args:
            text: входной текст
            use_few_shot: использовать few-shot примеры
            
        Returns:
            List[Dict]: список сущностей
        """
        # Ограничиваем длину текста
        if len(text) > 1000:
            text = text[:1000]
        
        # Получаем промпт
        prompt = get_ner_prompt(text, language=self.language, use_few_shot=use_few_shot)
        
        # Генерируем ответ
        response = self.generate_response(prompt)
        
        # Парсим JSON
        result = self.parse_json_response(response)
        
        return result.get("entities", [])
    
    def extract_entities_batch(
        self,
        texts: List[str],
        use_few_shot: bool = False,
        show_progress: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Извлечь сущности из списка текстов
        
        Args:
            texts: список текстов
            use_few_shot: использовать few-shot примеры
            show_progress: показывать прогресс-бар
            
        Returns:
            List[List[Dict]]: список списков сущностей
        """
        results = []
        
        iterator = tqdm(texts, desc="Извлечение сущностей") if show_progress else texts
        
        for text in iterator:
            entities = self.extract_entities(text, use_few_shot=use_few_shot)
            results.append(entities)
        
        return results


# Пример использования
if __name__ == "__main__":
    # Создаем NER модель
    ner = LLMNER(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cpu",  # или "cuda"
        language="ru"
    )
    
    # Тестовый текст
    text = "Владимир Путин посетил Москву 1 января 2023 года и встретился с представителями компании Яндекс."
    
    # Извлекаем сущности
    entities = ner.extract_entities(text)
    
    print("Найденные сущности:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['type']})")


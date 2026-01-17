"""
Тестовый скрипт для запуска NER с LLM на данных Википедии
"""

import sys
import os
import json

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner.llm_ner import LLMNER


def load_wikipedia_data(filepath, max_articles=5):
    """Загрузить данные из Wikipedia JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    return articles[:max_articles]


def main():
    print("=" * 70)
    print("Тестирование NER с локальными LLM")
    print("=" * 70)
    
    # Путь к данным
    data_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'processed',
        'wikipedia_articles.json'
    )
    
    # Проверка существования файла
    if not os.path.exists(data_path):
        print(f"Ошибка: файл не найден: {data_path}")
        print("Сначала запустите скрипт parse_wikipedia_dump.py")
        return
    
    # Загрузка данных
    print("\n1. Загрузка данных...")
    articles = load_wikipedia_data(data_path, max_articles=3)
    print(f"Загружено статей: {len(articles)}")
    
    # Создание NER модели
    print("\n2. Загрузка модели LLM...")
    print("Используется TinyLlama-1.1B (небольшая модель для теста)")
    print("Для лучшего качества используйте модели 7B (Mistral, LLaMA-2)")
    
    ner = LLMNER(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cpu",  # Измените на "cuda" если есть GPU
        language="ru"
    )
    
    # Тестирование на статьях
    print("\n3. Извлечение сущностей...")
    print("=" * 70)
    
    for i, article in enumerate(articles, 1):
        print(f"\nСтатья {i}: {article['title']}")
        print("-" * 70)
        
        # Берем первые 500 символов текста
        text_sample = article['text'][:500]
        print(f"Текст (первые 500 символов):")
        print(f"{text_sample}...")
        
        # Извлекаем сущности
        print(f"\nИзвлечение сущностей...")
        entities = ner.extract_entities(text_sample)
        
        if entities:
            print(f"\nНайдено сущностей: {len(entities)}")
            for entity in entities:
                print(f"  - {entity['text']:<30} ({entity['type']})")
        else:
            print("Сущности не найдены")
        
        print()
    
    print("=" * 70)
    print("Тестирование завершено!")
    print("\nПримечание:")
    print("- TinyLlama-1.1B - очень легкая модель, качество может быть низким")
    print("- Для лучших результатов используйте модели 7B:")
    print("  * mistralai/Mistral-7B-Instruct-v0.2")
    print("  * meta-llama/Llama-2-7b-chat-hf")
    print("  * ai-forever/saiga2_7b (для русского)")


if __name__ == "__main__":
    main()


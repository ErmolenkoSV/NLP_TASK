"""
Сравнение различных методов NER
"""

import sys
import os
import json
import time
from typing import List, Dict, Any
import numpy as np

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner.spacy_ner import SpacyNER
from src.ner.bert_ner import BERTNER
from src.ner.natasha_ner import NatashaNER


def load_wikipedia_data(filepath, max_articles=5):
    """Загрузить данные из Wikipedia JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    return articles[:max_articles]


def evaluate_ner_method(ner, name, articles, max_chars=500):
    """
    Оценка метода NER
    
    Args:
        ner: объект NER
        name: название метода
        articles: список статей
        max_chars: максимальное количество символов для обработки
        
    Returns:
        Dict: результаты
    """
    print(f"\n{'='*70}")
    print(f"Метод: {name}")
    print(f"{'='*70}")
    
    results = {
        "method": name,
        "total_articles": len(articles),
        "total_entities": 0,
        "avg_time_per_article": 0,
        "entities_by_type": {},
        "examples": []
    }
    
    total_time = 0
    
    for i, article in enumerate(articles, 1):
        print(f"\nСтатья {i}: {article['title']}")
        print("-" * 70)
        
        # Берем первые N символов
        text_sample = article['text'][:max_chars]
        
        # Измеряем время
        start_time = time.time()
        entities = ner.extract_entities(text_sample)
        elapsed_time = time.time() - start_time
        
        total_time += elapsed_time
        
        # Статистика
        results["total_entities"] += len(entities)
        
        for entity in entities:
            entity_type = entity["type"]
            results["entities_by_type"][entity_type] = \
                results["entities_by_type"].get(entity_type, 0) + 1
        
        # Выводим результаты
        print(f"Найдено сущностей: {len(entities)} (время: {elapsed_time:.2f}с)")
        
        if entities:
            print("Примеры:")
            for entity in entities[:10]:  # Первые 10
                print(f"  - {entity['text']:<30} ({entity['type']})")
        else:
            print("Сущности не найдены")
        
        # Сохраняем пример
        if i <= 2:  # Первые 2 статьи
            results["examples"].append({
                "title": article['title'],
                "entities": entities[:5]  # Первые 5 сущностей
            })
    
    results["avg_time_per_article"] = total_time / len(articles)
    
    # Итоговая статистика
    print(f"\n{'='*70}")
    print(f"Итоговая статистика для {name}:")
    print(f"  Обработано статей: {results['total_articles']}")
    print(f"  Всего сущностей: {results['total_entities']}")
    print(f"  Среднее время на статью: {results['avg_time_per_article']:.2f}с")
    print(f"  Сущности по типам:")
    for entity_type, count in sorted(results["entities_by_type"].items(), key=lambda x: x[1], reverse=True):
        print(f"    - {entity_type}: {count}")
    
    return results


def main():
    print("=" * 70)
    print("Сравнение методов NER")
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
        return
    
    # Загрузка данных
    print("\n1. Загрузка данных...")
    articles = load_wikipedia_data(data_path, max_articles=5)
    print(f"Загружено статей: {len(articles)}")
    
    # Список методов для тестирования
    all_results = []
    
    # 1. Natasha (быстрый и для русского)
    try:
        print("\n2. Загрузка Natasha...")
        ner_natasha = NatashaNER()
        results_natasha = evaluate_ner_method(ner_natasha, "Natasha", articles)
        all_results.append(results_natasha)
    except Exception as e:
        print(f"Ошибка Natasha: {e}")
    
    # 2. spaCy
    try:
        print("\n3. Загрузка spaCy...")
        ner_spacy = SpacyNER(model_name="ru_core_news_sm", language="ru")
        results_spacy = evaluate_ner_method(ner_spacy, "spaCy (ru_core_news_sm)", articles)
        all_results.append(results_spacy)
    except Exception as e:
        print(f"Ошибка spaCy: {e}")
        print("Установите модель: python -m spacy download ru_core_news_sm")
    
    # 3. BERT (английский, для сравнения)
    try:
        print("\n4. Загрузка BERT...")
        ner_bert = BERTNER(model_name="dslim/bert-base-NER", device="cpu")
        results_bert = evaluate_ner_method(ner_bert, "BERT (dslim/bert-base-NER)", articles)
        all_results.append(results_bert)
    except Exception as e:
        print(f"Ошибка BERT: {e}")
    
    # Сводная таблица сравнения
    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print("=" * 70)
    
    print(f"\n{'Метод':<30} {'Сущностей':<12} {'Время (с)':<12} {'Скорость'}")
    print("-" * 70)
    
    for result in all_results:
        entities_per_sec = result["total_entities"] / result["avg_time_per_article"] if result["avg_time_per_article"] > 0 else 0
        print(f"{result['method']:<30} {result['total_entities']:<12} {result['avg_time_per_article']:<12.2f} {entities_per_sec:.1f} сущ/с")
    
    # Сохранение результатов
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'evaluations', 'ner_comparison.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Преобразуем результаты в сериализуемый формат (конвертируем numpy/tensor типы)
    def to_builtin_types(obj):
        if isinstance(obj, dict):
            return {k: to_builtin_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_builtin_types(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Пробуем стандартные типы (float32 из torch/np может попадать сюда)
        try:
            # Проверяем на совместимые методы
            if hasattr(obj, "item"):
                return obj.item()
        except Exception:
            pass
        return obj

    serializable_results = to_builtin_types(all_results)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в: {output_path}")


if __name__ == "__main__":
    main()


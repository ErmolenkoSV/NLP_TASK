"""
Скрипт для распаковки и парсинга дампа Википедии
Извлекает тексты статей из .bz2 файла и сохраняет в JSON
"""

import bz2
import xml.etree.ElementTree as ET
import json
import os
from tqdm import tqdm
import re

def clean_wiki_text(text):
    """Очистка вики-разметки из текста"""
    # Удаление вики-разметки
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)  # [[link|text]] -> text
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[link]] -> link
    text = re.sub(r'\{\{.*?\}\}', '', text)  # {{templates}}
    text = re.sub(r'<ref>.*?</ref>', '', text, flags=re.DOTALL)  # <ref>...</ref>
    text = re.sub(r'<[^>]+>', '', text)  # HTML теги
    text = re.sub(r'={2,}.*?={2,}', '', text)  # Заголовки ==title==
    text = re.sub(r"''+", '', text)  # Курсив/жирный
    text = re.sub(r'\n+', ' ', text)  # Множественные переносы
    text = re.sub(r'\s+', ' ', text)  # Множественные пробелы
    return text.strip()

def parse_wikipedia_dump(bz2_file, output_file, max_articles=None, min_text_length=100):
    """
    Парсит дамп Википедии и извлекает тексты статей
    
    Args:
        bz2_file: путь к .bz2 файлу
        output_file: путь к выходному JSON файлу
        max_articles: максимальное количество статей (None = все)
        min_text_length: минимальная длина текста статьи
    """
    articles = []
    
    print(f"Распаковываю и парсю файл: {bz2_file}")
    print(f"Максимум статей: {max_articles if max_articles else 'все'}")
    
    try:
        with bz2.open(bz2_file, 'rb') as f:
            # Используем iterparse для обработки больших XML файлов
            context = ET.iterparse(f, events=('end',))
            
            current_title = None
            current_text = None
            current_ns = None
            article_count = 0
            page_count = 0
            
            print("Обработка статей...")
            
            for event, elem in tqdm(context, desc="Парсинг"):
                tag = elem.tag
                
                # Получаем имя тега без namespace
                if '}' in tag:
                    tag = tag.split('}')[1]
                
                if tag == 'page':
                    # Обработка завершенной страницы
                    page_count += 1
                    
                    if current_text and len(current_text) >= min_text_length:
                        # Проверяем namespace (0 = обычная статья)
                        if current_ns == '0' or current_ns == 0:
                            cleaned_text = clean_wiki_text(current_text[:50000])
                            if len(cleaned_text) >= min_text_length:
                                articles.append({
                                    'title': current_title or '',
                                    'text': cleaned_text
                                })
                                article_count += 1
                                
                                if max_articles and article_count >= max_articles:
                                    break
                    
                    # Сброс переменных
                    current_title = None
                    current_text = None
                    current_ns = None
                    
                    # Очистка памяти
                    elem.clear()
                
                elif tag == 'title':
                    if elem.text:
                        current_title = elem.text
                
                elif tag == 'ns':
                    if elem.text:
                        current_ns = elem.text.strip()
                
                elif tag == 'text':
                    if elem.text:
                        current_text = elem.text
        
        print(f"\nИзвлечено статей: {len(articles)}")
        
        # Сохранение в JSON
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Результаты сохранены в: {output_file}")
        print(f"Общий размер текста: {sum(len(a['text']) for a in articles) / 1024 / 1024:.2f} MB")
        
        return articles
    
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    # Пути к файлам
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'ruwiki-latest-pages-articles4.xml-p3698270p3835772.bz2')
    output_file = os.path.join(project_dir, 'data', 'processed', 'wikipedia_articles.json')
    
    # Проверка существования входного файла
    if not os.path.exists(input_file):
        print(f"Ошибка: файл не найден: {input_file}")
        print("Пожалуйста, убедитесь, что файл находится в корне проекта task3")
        return
    
    # Парсинг (первые 100 статей для теста, можно изменить)
    articles = parse_wikipedia_dump(
        bz2_file=input_file,
        output_file=output_file,
        max_articles=100,  # Изменить на None для всех статей
        min_text_length=200
    )
    
    if articles:
        print("\nПримеры извлеченных статей:")
        print("-" * 70)
        for i, article in enumerate(articles[:5], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Длина текста: {len(article['text'])} символов")
            print(f"   Начало: {article['text'][:200]}...")

if __name__ == '__main__':
    main()


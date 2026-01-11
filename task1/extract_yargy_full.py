"""
Извлечение структурированной информации из новостных статей
с использованием библиотеки Yargy.
"""

import gzip
import re
from dataclasses import dataclass
from typing import Optional, List
from yargy import Parser, rule
from yargy.predicates import gram
from yargy.interpretation import fact


@dataclass
class Entry:
    name: str
    birth_date: Optional[str] = None
    birth_place: Optional[str] = None


# Определяем структуру для Yargy
Person = fact('Person', ['first', 'middle', 'last'])

# Грамматические правила
NAME_FULL = rule(
    gram('Name').interpretation(Person.first),
    gram('Patr').interpretation(Person.middle),
    gram('Surn').interpretation(Person.last)
).interpretation(Person)

NAME_SHORT = rule(
    gram('Name').interpretation(Person.first),
    gram('Surn').interpretation(Person.last)
).interpretation(Person)


def extract_names_with_yargy(text: str) -> List[str]:
    names = []
    
    try:
        # Полные имена
        parser_full = Parser(NAME_FULL)
        for match in parser_full.findall(text):
            person = match.fact
            name_parts = []
            if person.first:
                name_parts.append(person.first)
            if person.middle:
                name_parts.append(person.middle)
            if person.last:
                name_parts.append(person.last)
            
            if len(name_parts) >= 2:
                full_name = ' '.join(name_parts)
                if is_valid_name(full_name) and full_name not in names:
                    names.append(full_name)
        
        # Короткие имена
        parser_short = Parser(NAME_SHORT)
        for match in parser_short.findall(text):
            person = match.fact
            if person.first and person.last:
                full_name = f"{person.first} {person.last}"
                if not any(full_name in existing for existing in names):
                    if is_valid_name(full_name) and full_name not in names:
                        names.append(full_name)
    
    except Exception:
        pass
    
    return names[:4]


def is_valid_name(name: str) -> bool:
    words = name.split()
    
    stop_words = {
        'Президент', 'Министр', 'Губернатор', 'Депутат', 'Сенатор',
        'Адмирал', 'Генерал', 'Полковник', 'Майор', 'Капитан',
        'России', 'Российской', 'Москвы', 'Украины', 'США', 'Европы',
        'Сбербанка', 'Газпрома', 'Роснефти', 'ВТБ', 'Лукойла',
        'Союза', 'Федерации', 'Республики', 'Области', 'Края',
        'Группа', 'Компания', 'Корпорация', 'Холдинг', 'Концерн',
        'Академии', 'Университета', 'Института', 'Школы', 'Центра',
        'Театра', 'Музея', 'Галереи', 'Студии', 'Дома',
        'Банка', 'Фонда', 'Агентства', 'Службы', 'Департамента'
    }
    
    if any(word in stop_words for word in words):
        return False
    
    if len(words) < 2 or len(words) > 3:
        return False
    
    surname_endings = ['ов', 'ев', 'ин', 'ын', 'ский', 'цкий', 'ова', 'ева', 'ина', 'ская', 'цкая']
    has_surname = any(word.endswith(end) for word in words for end in surname_endings)
    
    if not has_surname:
        return False
    
    if any(char.isdigit() for char in name):
        return False
    
    return True


def extract_birth_dates(text: str) -> List[str]:
    dates = []
    
    if not any(word in text.lower() for word in ['родился', 'родилась', 'рождения', 'родившийся', 'родившаяся']):
        return dates
    
    months = {
        'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
        'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
        'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
    }
    
    pattern1 = r'родил(?:ся|ась|ся в|ась в)\s+(\d{1,2})\s+(' + '|'.join(months.keys()) + r')\s+(\d{4})'
    
    for match in re.finditer(pattern1, text, re.IGNORECASE):
        day = match.group(1).zfill(2)
        month = months[match.group(2).lower()]
        year = match.group(3)
        
        try:
            year_int = int(year)
            if 1850 <= year_int <= 2024:
                date_str = f"{day}.{month}.{year}"
                if date_str not in dates:
                    dates.append(date_str)
        except:
            pass
    
    pattern2 = r'(?:дата рождения|родился|родилась)[:\s]+(\d{1,2})\.(\d{1,2})\.(\d{4})'
    
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        day = match.group(1).zfill(2)
        month = match.group(2).zfill(2)
        year = match.group(3)
        
        try:
            year_int = int(year)
            if 1850 <= year_int <= 2024:
                date_str = f"{day}.{month}.{year}"
                if date_str not in dates:
                    dates.append(date_str)
        except:
            pass
    
    return dates


def extract_birth_places(text: str) -> List[str]:
    places = []
    
    if not any(word in text.lower() for word in ['родился', 'родилась', 'рождения']):
        return places
    
    cities = [
        'Москва', 'Москве', 'Москвы',
        'Санкт-Петербург', 'Санкт-Петербурге', 'Петербург', 'Петербурге',
        'Новосибирск', 'Новосибирске',
        'Екатеринбург', 'Екатеринбурге',
        'Казань', 'Казани',
        'Нижний Новгород', 'Нижнем Новгороде',
        'Ростов-на-Дону', 'Ростове-на-Дону',
        'Уфа', 'Уфе',
        'Красноярск', 'Красноярске',
        'Воронеж', 'Воронеже',
        'Пермь', 'Перми',
        'Волгоград', 'Волгограде',
        'Краснодар', 'Краснодаре',
        'Саратов', 'Саратове',
        'Тюмень', 'Тюмени',
        'Омск', 'Омске',
        'Самара', 'Самаре'
    ]
    
    for city in cities:
        pattern = rf'родил(?:ся|ась)\s+в\s+({re.escape(city)})'
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            place = match.group(1)
            if place.lower() in ['москве', 'москвы']:
                place = 'Москва'
            elif place.lower() in ['петербурге', 'санкт-петербурге']:
                place = 'Санкт-Петербург'
            elif place.lower() in ['ростове-на-дону']:
                place = 'Ростов-на-Дону'
            
            if place not in places:
                places.append(place)
                break
    
    return places


def process_news_file(filepath: str, max_articles: int = 5000) -> List[Entry]:
    entries = []
    processed = 0
    
    print(f"Файл: {filepath}")
    print(f"Обрабатываю до {max_articles} статей")
    print(f"Использую Yargy для извлечения имен...")
    print()
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            if processed >= max_articles:
                break
            
            try:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                category, title, text = parts[0], parts[1], parts[2]
                full_text = title + ' ' + text
                
                names = extract_names_with_yargy(full_text)
                dates = extract_birth_dates(full_text)
                places = extract_birth_places(full_text)
                
                if names:
                    for name in names:
                        entry = Entry(
                            name=name,
                            birth_date=dates[0] if dates else None,
                            birth_place=places[0] if places else None
                        )
                        entries.append(entry)
                
                processed += 1
                
                if processed % 500 == 0:
                    print(f"Обработано: {processed:>4} | Найдено: {len(entries):>4}")
                    
            except Exception:
                continue
    
    print(f"\nВсего обработано: {processed}")
    print(f"Найдено записей: {len(entries)}")
    
    return entries


def main():
    print("=" * 70)
    print("Извлечение структурированной информации из новостей")
    print("Библиотека: Yargy")
    print("=" * 70)
    print()
    
    entries = process_news_file('news.txt.gz', max_articles=5000)
    
    complete_entries = [e for e in entries if e.birth_date or e.birth_place]
    simple_entries = [e for e in entries if not e.birth_date and not e.birth_place]
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 70)
    
    if complete_entries:
        print("\n>>> С датами или местами рождения:")
        print("-" * 70)
        for i, e in enumerate(complete_entries[:25], 1):
            print(f"\n{i}. Имя: {e.name}")
            if e.birth_date:
                print(f"   Дата рождения: {e.birth_date}")
            if e.birth_place:
                print(f"   Место рождения: {e.birth_place}")
    
    if simple_entries:
        print("\n\n>>> Извлеченные имена:")
        print("-" * 70)
        for i, e in enumerate(simple_entries[:50], 1):
            print(f"{i}. {e.name}")
    
    output_file = 'extracted_yargy_full.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Результаты извлечения\n")
        f.write("=" * 60 + "\n\n")
        
        for entry in entries:
            f.write(f"Имя: {entry.name}\n")
            if entry.birth_date:
                f.write(f"Дата рождения: {entry.birth_date}\n")
            if entry.birth_place:
                f.write(f"Место рождения: {entry.birth_place}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\n\nСохранено: {output_file}")
    
    with_dates = sum(1 for e in entries if e.birth_date)
    with_places = sum(1 for e in entries if e.birth_place)
    complete = sum(1 for e in entries if e.birth_date and e.birth_place)
    
    print("\n" + "=" * 70)
    print("Статистика:")
    print("=" * 70)
    print(f"  Всего имен:     {len(entries)}")
    print(f"  С датами:       {with_dates} ({with_dates/len(entries)*100:.1f}%)")
    print(f"  С местами:      {with_places} ({with_places/len(entries)*100:.1f}%)")
    print(f"  Полных записей: {complete}")
    print("=" * 70)


if __name__ == '__main__':
    main()

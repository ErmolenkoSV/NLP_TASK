"""
Промпты для извлечения именованных сущностей с помощью LLM
"""

# Промпт для извлечения сущностей (русский язык)
NER_PROMPT_RU = """Извлеки все именованные сущности из текста и верни результат в формате JSON.

Текст: {text}

Извлеки следующие типы сущностей:
- PERSON: Имена людей
- ORG: Организации, компании, учреждения
- LOC: Географические объекты (города, страны, регионы)
- DATE: Даты и временные периоды
- MONEY: Денежные суммы
- PRODUCT: Названия продуктов, товаров
- EVENT: Названия событий

Верни результат СТРОГО в следующем JSON формате (без дополнительного текста):
{{
  "entities": [
    {{
      "text": "название сущности",
      "type": "PERSON|ORG|LOC|DATE|MONEY|PRODUCT|EVENT",
      "start": позиция_начала_в_тексте,
      "end": позиция_конца_в_тексте
    }}
  ]
}}

JSON:"""

# Промпт для извлечения сущностей (английский язык)
NER_PROMPT_EN = """Extract all named entities from the text and return the result in JSON format.

Text: {text}

Extract the following entity types:
- PERSON: Names of people
- ORG: Organizations, companies, institutions
- LOC: Geographic locations (cities, countries, regions)
- DATE: Dates and time periods
- MONEY: Monetary amounts
- PRODUCT: Product names
- EVENT: Event names

Return the result STRICTLY in the following JSON format (without additional text):
{{
  "entities": [
    {{
      "text": "entity text",
      "type": "PERSON|ORG|LOC|DATE|MONEY|PRODUCT|EVENT",
      "start": start_position_in_text,
      "end": end_position_in_text
    }}
  ]
}}

JSON:"""

# Промпт для извлечения отношений между сущностями
RELATION_EXTRACTION_PROMPT_RU = """Найди отношения между сущностями в тексте.

Текст: {text}

Найденные сущности:
{entities}

Определи отношения между парами сущностей и верни в JSON формате:
{{
  "relations": [
    {{
      "entity1": "сущность1",
      "relation": "тип_отношения",
      "entity2": "сущность2"
    }}
  ]
}}

Доступные типы отношений:
- работает_в: человек работает в организации
- основал: человек/организация основала организацию
- родился_в: человек родился в локации
- находится_в: организация/событие находится в локации
- происходил_в: событие происходило в локации/дате
- руководит: человек руководит организацией
- участвовал_в: человек участвовал в событии

JSON:"""

# Few-shot примеры для улучшения качества
FEW_SHOT_EXAMPLES_RU = """
Пример 1:
Текст: "Владимир Путин посетил Москву 1 января 2023 года."
JSON:
{{
  "entities": [
    {{"text": "Владимир Путин", "type": "PERSON", "start": 0, "end": 14}},
    {{"text": "Москву", "type": "LOC", "start": 24, "end": 30}},
    {{"text": "1 января 2023 года", "type": "DATE", "start": 31, "end": 49}}
  ]
}}

Пример 2:
Текст: "Яндекс открыл новый офис в Санкт-Петербурге."
JSON:
{{
  "entities": [
    {{"text": "Яндекс", "type": "ORG", "start": 0, "end": 6}},
    {{"text": "Санкт-Петербурге", "type": "LOC", "start": 28, "end": 44}}
  ]
}}

Теперь извлеки сущности из следующего текста:
"""

def get_ner_prompt(text, language="ru", use_few_shot=False):
    """
    Получить промпт для NER
    
    Args:
        text: текст для обработки
        language: язык ("ru" или "en")
        use_few_shot: использовать few-shot примеры
        
    Returns:
        str: готовый промпт
    """
    if language == "ru":
        base_prompt = NER_PROMPT_RU
        few_shot = FEW_SHOT_EXAMPLES_RU if use_few_shot else ""
    else:
        base_prompt = NER_PROMPT_EN
        few_shot = ""
    
    if use_few_shot:
        return few_shot + "\n" + base_prompt.format(text=text)
    else:
        return base_prompt.format(text=text)

def get_relation_extraction_prompt(text, entities, language="ru"):
    """
    Получить промпт для извлечения отношений
    
    Args:
        text: текст для обработки
        entities: список найденных сущностей
        language: язык ("ru" или "en")
        
    Returns:
        str: готовый промпт
    """
    entities_str = "\n".join([f"- {e['text']} ({e['type']})" for e in entities])
    
    if language == "ru":
        return RELATION_EXTRACTION_PROMPT_RU.format(text=text, entities=entities_str)
    else:
        # TODO: добавить английский промпт
        return RELATION_EXTRACTION_PROMPT_RU.format(text=text, entities=entities_str)


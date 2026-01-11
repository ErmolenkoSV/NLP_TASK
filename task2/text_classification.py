"""
Классификация новостных текстов с помощью Word2Vec и SVM.

Нужно обучить Word2Vec, преобразовать документы в векторы разными способами
и сравнить результаты классификации.
"""

import gzip
import re
import numpy as np
from collections import Counter
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Пробуем подключить pymorphy2 для нормализации, но не обязательно
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    PYMPHY2_AVAILABLE = True
except (ImportError, AttributeError):
    PYMPHY2_AVAILABLE = False
    print("Внимание: pymorphy2 недоступен, нормализация отключена")

# Список стоп-слов для фильтрации
STOP_WORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его',
    'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
    'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже',
    'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
    'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их',
    'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда',
    'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти',
    'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец',
    'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них',
    'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
    'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
}


def preprocess_text(text: str, normalize: bool = True, remove_stopwords: bool = True) -> List[str]:
    """Обработка текста перед обучением"""
    # Убираем все кроме букв
    text = re.sub(r'[^а-яёa-z\s]', ' ', text.lower())
    words = text.split()
    
    # Нормализуем если есть pymorphy2
    if normalize and PYMPHY2_AVAILABLE:
        normalized_words = []
        for word in words:
            if len(word) > 2:
                try:
                    parsed = morph.parse(word)[0]
                    normalized_words.append(parsed.normal_form)
                except:
                    normalized_words.append(word)
        words = normalized_words
    elif normalize and not PYMPHY2_AVAILABLE:
        words = [w for w in words if len(w) > 2]
    
    # Убираем стоп-слова
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    return words


def load_news_data(filepath: str, max_articles: int = 10000) -> Tuple[List[str], List[str]]:
    """Читаем новости из архива"""
    texts = []
    labels = []
    
    print(f"Загрузка данных из {filepath}...")
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_articles:
                break
            
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    category = parts[0]
                    title = parts[1]
                    text = parts[2]
                    
                    # Объединяем заголовок и текст
                    full_text = title + ' ' + text
                    texts.append(full_text)
                    labels.append(category)
                    
                if (i + 1) % 1000 == 0:
                    print(f"Загружено: {i + 1} статей")
                    
            except Exception:
                continue
    
    print(f"Всего загружено: {len(texts)} статей")
    print(f"Категории: {set(labels)}")
    
    return texts, labels


def train_word2vec(texts: List[str], vector_size: int = 100, window: int = 5) -> Word2Vec:
    """Обучаем Word2Vec на наших текстах"""
    print("\nОбработка текстов для Word2Vec...")
    
    use_normalize = PYMPHY2_AVAILABLE
    processed_texts = []
    for text in texts:
        words = preprocess_text(text, normalize=use_normalize, remove_stopwords=True)
        if len(words) > 0:
            processed_texts.append(words)
    
    print(f"Обработано текстов: {len(processed_texts)}")
    if not PYMPHY2_AVAILABLE:
        print("Нормализация отключена (pymorphy2 недоступен)")
    print(f"Обучаю модель Word2Vec...")
    
    # Обучаем Word2Vec
    model = Word2Vec(
        sentences=processed_texts,
        vector_size=vector_size,
        window=window,
        min_count=2,
        workers=4,
        sg=0  # CBOW алгоритм
    )
    
    print(f"Модель обучена! Размер словаря: {len(model.wv)}")
    
    return model, processed_texts


def document_to_vector_simple(words: List[str], model: Word2Vec) -> np.ndarray:
    """Просто берем среднее всех векторов слов"""
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)


def document_to_vector_weighted(words: List[str], model: Word2Vec) -> np.ndarray:
    """Взвешиваем по частоте слова в документе"""
    if len(words) == 0:
        return np.zeros(model.vector_size)
    
    word_freq = Counter(words)
    total_words = len(words)
    
    weighted_vectors = []
    for word, freq in word_freq.items():
        if word in model.wv:
            weight = freq / total_words
            weighted_vectors.append(model.wv[word] * weight)
    
    if len(weighted_vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.sum(weighted_vectors, axis=0)


def document_to_vector_max_pooling(words: List[str], model: Word2Vec) -> np.ndarray:
    """Max-pooling: берем максимум по каждой размерности"""
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    vectors_array = np.array(vectors)
    return np.max(vectors_array, axis=0)


def document_to_vector_tfidf_like(words: List[str], model: Word2Vec, 
                                   document_frequencies: dict, total_docs: int) -> np.ndarray:
    """TF-IDF взвешивание: учитываем важность слова в документе и коллекции"""
    if len(words) == 0:
        return np.zeros(model.vector_size)
    
    word_freq = Counter(words)
    total_words = len(words)
    
    weighted_vectors = []
    for word, freq in word_freq.items():
        if word in model.wv:
            # Частота слова в документе
            tf = freq / total_words
            
            # Обратная частота документа
            if word in document_frequencies:
                idf = np.log(total_docs / (document_frequencies[word] + 1))
            else:
                idf = 1.0
            
            weight = tf * idf
            weighted_vectors.append(model.wv[word] * weight)
    
    if len(weighted_vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.sum(weighted_vectors, axis=0)


def texts_to_vectors(texts_words: List[List[str]], model: Word2Vec, 
                     method: str = 'simple') -> np.ndarray:
    """Преобразуем все тексты в векторы выбранным методом"""
    print(f"\nПреобразование текстов в векторы (метод: {method})...")
    total = len(texts_words)
    
    if method == 'simple':
        vectors = []
        for i, words in enumerate(texts_words):
            vectors.append(document_to_vector_simple(words, model))
            if (i + 1) % 1000 == 0:
                print(f"  Обработано: {i+1}/{total}")
    elif method == 'weighted':
        vectors = []
        for i, words in enumerate(texts_words):
            vectors.append(document_to_vector_weighted(words, model))
            if (i + 1) % 1000 == 0:
                print(f"  Обработано: {i+1}/{total}")
    elif method == 'max_pooling':
        vectors = []
        for i, words in enumerate(texts_words):
            vectors.append(document_to_vector_max_pooling(words, model))
            if (i + 1) % 1000 == 0:
                print(f"  Обработано: {i+1}/{total}")
    elif method == 'tfidf':
        # Считаем в скольких документах встречается каждое слово
        print("  Подсчет частот документов...")
        document_frequencies = {}
        for words in texts_words:
            unique_words = set(words)
            for word in unique_words:
                document_frequencies[word] = document_frequencies.get(word, 0) + 1
        
        total_docs = len(texts_words)
        vectors = []
        for i, words in enumerate(texts_words):
            vectors.append(document_to_vector_tfidf_like(words, model, document_frequencies, total_docs))
            if (i + 1) % 1000 == 0:
                print(f"  Обработано: {i+1}/{total}")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.array(vectors)


def train_classifier(X_train, y_train, X_test, y_test, method_name: str):
    """Обучаем SVM и смотрим результаты"""
    print(f"\nОбучение SVM (метод: {method_name})...")
    
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nТочность ({method_name}): {accuracy:.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred))
    
    return classifier, accuracy


def main():
    """Главная функция - запускаем весь процесс"""
    print("=" * 70)
    print("КЛАССИФИКАЦИЯ ТЕКСТОВ С ВЕКТОРНЫМИ ПРЕДСТАВЛЕНИЯМИ СЛОВ")
    print("=" * 70)
    
    # Загружаем данные
    filepath = '../task1/news.txt.gz'
    texts, labels = load_news_data(filepath, max_articles=5000)
    
    # Обучаем Word2Vec
    model, processed_texts = train_word2vec(texts, vector_size=100, window=5)
    
    # Делим на train/test
    print("\nРазделение на обучающую и тестовую выборки...")
    X_train_words, X_test_words, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Обучающая выборка: {len(X_train_words)}")
    print(f"Тестовая выборка: {len(X_test_words)}")
    
    # Тестируем разные методы
    print("\n" + "=" * 70)
    print("МЕТОД 1: ПРОСТОЕ УСРЕДНЕНИЕ ВЕКТОРОВ")
    print("=" * 70)
    
    X_train_simple = texts_to_vectors(X_train_words, model, method='simple')
    X_test_simple = texts_to_vectors(X_test_words, model, method='simple')
    classifier1, accuracy1 = train_classifier(X_train_simple, y_train, X_test_simple, y_test, 'Простое усреднение')
    
    print("\n" + "=" * 70)
    print("МЕТОД 2: ВЗВЕШЕННОЕ УСРЕДНЕНИЕ ПО ЧАСТОТЕ")
    print("=" * 70)
    
    X_train_weighted = texts_to_vectors(X_train_words, model, method='weighted')
    X_test_weighted = texts_to_vectors(X_test_words, model, method='weighted')
    classifier2, accuracy2 = train_classifier(X_train_weighted, y_train, X_test_weighted, y_test, 'Взвешенное усреднение')
    
    print("\n" + "=" * 70)
    print("МЕТОД 3: MAX-POOLING")
    print("=" * 70)
    
    X_train_max = texts_to_vectors(X_train_words, model, method='max_pooling')
    X_test_max = texts_to_vectors(X_test_words, model, method='max_pooling')
    classifier3, accuracy3 = train_classifier(X_train_max, y_train, X_test_max, y_test, 'Max-pooling')
    
    print("\n" + "=" * 70)
    print("МЕТОД 4: TF-IDF ПОДОБНОЕ ВЗВЕШИВАНИЕ")
    print("=" * 70)
    
    X_train_tfidf = texts_to_vectors(X_train_words, model, method='tfidf')
    X_test_tfidf = texts_to_vectors(X_test_words, model, method='tfidf')
    classifier4, accuracy4 = train_classifier(X_train_tfidf, y_train, X_test_tfidf, y_test, 'TF-IDF взвешивание')
    
    # Сравниваем результаты
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 70)
    
    results = [
        ('Простое усреднение', accuracy1),
        ('Взвешенное усреднение', accuracy2),
        ('Max-pooling', accuracy3),
        ('TF-IDF взвешивание', accuracy4)
    ]
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\nРейтинг методов (по точности):")
    for i, (method, acc) in enumerate(results, 1):
        print(f"{i}. {method}: {acc:.4f} ({acc*100:.2f}%)")
    
    model.save('word2vec_model.bin')
    print("\nМодель Word2Vec сохранена в word2vec_model.bin")
    
    print("\n" + "=" * 70)
    print("ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == '__main__':
    main()


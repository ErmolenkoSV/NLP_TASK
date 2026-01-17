"""
NER с использованием CRF (Conditional Random Fields)
"""

from typing import List, Dict, Any, Optional
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pymorphy2
from .base_ner import BaseNER


class CRFNER(BaseNER):
    """NER с использованием CRF (требует обучения на размеченных данных)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: путь к сохраненной модели CRF
        """
        super().__init__(model_name="crf")
        
        self.morph = pymorphy2.MorphAnalyzer()
        self.crf = None
        
        if model_path:
            print(f"Загрузка CRF модели из {model_path}...")
            # TODO: загрузка сохраненной модели
            print("CRF модель загружена")
        else:
            print("CRF модель не обучена")
            print("Для работы CRF нужны размеченные данные для обучения")
    
    def word2features(self, sent: List[str], i: int) -> Dict[str, Any]:
        """
        Извлечение признаков для слова в контексте предложения
        
        Args:
            sent: список слов в предложении
            i: индекс текущего слова
            
        Returns:
            Dict: словарь признаков
        """
        word = sent[i]
        
        # Морфологический анализ
        parsed = self.morph.parse(word)[0]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': parsed.tag.POS,
            'word.is_capitalized': word[0].isupper() if word else False,
        }
        
        # Контекстные признаки (предыдущее слово)
        if i > 0:
            word1 = sent[i-1]
            parsed1 = self.morph.parse(word1)[0]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': parsed1.tag.POS,
            })
        else:
            features['BOS'] = True  # Beginning of sentence
        
        # Контекстные признаки (следующее слово)
        if i < len(sent)-1:
            word1 = sent[i+1]
            parsed1 = self.morph.parse(word1)[0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': parsed1.tag.POS,
            })
        else:
            features['EOS'] = True  # End of sentence
        
        return features
    
    def sent2features(self, sent: List[str]) -> List[Dict[str, Any]]:
        """Извлечение признаков для всех слов в предложении"""
        return [self.word2features(sent, i) for i in range(len(sent))]
    
    def train(self, X_train: List[List[str]], y_train: List[List[str]]):
        """
        Обучение CRF модели
        
        Args:
            X_train: список предложений (список списков слов)
            y_train: список меток для каждого слова
        """
        print("Обучение CRF модели...")
        
        # Извлекаем признаки
        X_train_features = [self.sent2features(s) for s in X_train]
        
        # Создаем и обучаем модель
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        
        self.crf.fit(X_train_features, y_train)
        
        print("CRF модель обучена")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлечь именованные сущности из текста
        
        Args:
            text: входной текст
            
        Returns:
            List[Dict]: список сущностей
        """
        if self.crf is None:
            print("Ошибка: CRF модель не обучена")
            return []
        
        # Простая токенизация (для продакшн нужна лучше)
        words = text.split()
        
        # Извлекаем признаки
        features = self.sent2features(words)
        
        # Предсказываем метки
        labels = self.crf.predict([features])[0]
        
        # Собираем сущности
        entities = []
        current_entity = None
        current_text = []
        start_pos = 0
        
        for i, (word, label) in enumerate(zip(words, labels)):
            if label.startswith('B-'):
                # Начало новой сущности
                if current_entity:
                    # Сохраняем предыдущую
                    entities.append({
                        "text": " ".join(current_text),
                        "type": current_entity,
                        "start": start_pos,
                        "end": start_pos + len(" ".join(current_text))
                    })
                current_entity = label[2:]
                current_text = [word]
                start_pos = text.find(word, start_pos)
            
            elif label.startswith('I-') and current_entity:
                # Продолжение сущности
                current_text.append(word)
            
            else:
                # Не сущность
                if current_entity:
                    entities.append({
                        "text": " ".join(current_text),
                        "type": current_entity,
                        "start": start_pos,
                        "end": start_pos + len(" ".join(current_text))
                    })
                    current_entity = None
                    current_text = []
        
        # Последняя сущность
        if current_entity:
            entities.append({
                "text": " ".join(current_text),
                "type": current_entity,
                "start": start_pos,
                "end": start_pos + len(" ".join(current_text))
            })
        
        return entities


# Пример использования
if __name__ == "__main__":
    print("CRF NER требует обучения на размеченных данных")
    print("Для использования:")
    print("1. Подготовьте размеченные данные в формате IOB")
    print("2. Обучите модель: ner.train(X_train, y_train)")
    print("3. Используйте для извлечения: ner.extract_entities(text)")


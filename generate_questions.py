#!/usr/bin/env python3
"""
скрипт для генерации вопросов через API Кими
запуск: python generate_questions.py
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# конфиг
KIMI_API_KEY = os.environ.get('KIMI_API_KEY', '')
KIMI_API_URL = os.environ.get('KIMI_API_URL', 'https://api.moonshot.cn/v1/chat/completions')

# темы
TOPICS = [
    'история',
    'наука',
    'география',
    'спорт',
    'кино и музыка',
    'технологии',
    'литература',
    'биология',
    'космос',
    'видеоигры'
]

DIFFICULTIES = ['easy', 'medium', 'hard']


def generate_questions(topic, difficulty, count=35):
    """генерация вопросов через API Кими"""
    
    if not KIMI_API_KEY:
        print(f"[!] ошибка: нет KIMI_API_KEY")
        return None
    
    prompt = f"""создай {count} вопросов для викторины на тему "{topic}" с уровнем сложности "{difficulty}" (на русском языке).

требования:
- вопросы должны быть интересными и разнообразными
- 4 варианта ответа на каждый вопрос
- только один правильный ответ
- не используй markdown, только plain text

ответь СТРОГО в формате JSON без пояснений:
{{
  "questions": [
    {{
      "question": "текст вопроса",
      "options": ["вариант 1", "вариант 2", "вариант 3", "вариант 4"],
      "correct": 0
    }}
  ]
}}

gде correct - индекс (0-3) правильного ответа."""

    try:
        headers = {
            'Authorization': f'Bearer {KIMI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'moonshot-v1-8k',
            'messages': [
                {'role': 'system', 'content': 'ты - генератор вопросов для викторины. отвечай только в формате json без markdown.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 4000
        }
        
        print(f"[*] отправка запроса для темы '{topic}' ({difficulty})...")
        response = requests.post(KIMI_API_URL, headers=headers, json=data, timeout=120)
        
        if response.status_code != 200:
            print(f"[!] ошибка API: {response.status_code}")
            print(response.text)
            return None
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # парсим json
        try:
            # ищем json блок
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start == -1 or end == 0:
                print(f"[!] не найден JSON в ответе")
                print(f"[!] ответ: {content[:200]}")
                return None
            
            json_str = content[start:end]
            parsed = json.loads(json_str)
            questions = parsed.get('questions', [])
            
            # валидация
            valid_questions = []
            for q in questions:
                if all(k in q for k in ['question', 'options', 'correct']):
                    if len(q['options']) == 4 and 0 <= q['correct'] <= 3:
                        valid_questions.append(q)
            
            print(f"[+] получено {len(valid_questions)} валидных вопросов")
            return valid_questions
            
        except json.JSONDecodeError as e:
            print(f"[!] ошибка парсинга JSON: {e}")
            print(f"[!] контент: {content[:200]}")
            return None
            
    except Exception as e:
        print(f"[!] ошибка: {e}")
        return None


def save_to_db(questions, topic, difficulty):
    """сохранение вопросов в базу данных"""
    from app import app, db, Question
    
    with app.app_context():
        for q in questions:
            try:
                question = Question(
                    topic=topic,
                    difficulty=difficulty,
                    question_text=q['question'],
                    option_1=q['options'][0],
                    option_2=q['options'][1],
                    option_3=q['options'][2],
                    option_4=q['options'][3],
                    correct_answer=q['correct']
                )
                db.session.add(question)
            except Exception as e:
                print(f"[!] ошибка сохранения вопроса: {e}")
                continue
        
        db.session.commit()
        print(f"[+] вопросы сохранены в базу данных")


def main():
    """основная функция"""
    print("=" * 50)
    print("генератор вопросов для quizbattle")
    print("=" * 50)
    
    if not KIMI_API_KEY:
        print("[!] укажите KIMI_API_KEY в .env файле")
        return
    
    total_generated = 0
    
    for topic in TOPICS:
        for difficulty in DIFFICULTIES:
            print(f"\n[*] тема: {topic}, сложность: {difficulty}")
            
            questions = generate_questions(topic, difficulty, 35)
            
            if questions and len(questions) > 0:
                save_to_db(questions, topic, difficulty)
                total_generated += len(questions)
            else:
                print(f"[!] не удалось сгенерировать вопросы")
            
            # небольшая пауза чтобы не перегружать API
            time.sleep(2)
    
    print("\n" + "=" * 50)
    print(f"всего сгенерировано: {total_generated} вопросов")
    print("=" * 50)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""скрипт для генерации вопросов через gemini api
запуск: python generate_questions.py

сценарий:
- генерирует пачку вопросов по темам/сложностям
- сохраняет в sqlite (через модели из app.py)

нужно:
- добавить GEMINI_API_KEY в .env
"""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv

from gemini_client import GeminiClient, extract_text

load_dotenv()

# конфиг
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro").strip()
GEMINI_API_BASE_URL = os.environ.get("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com").strip()

# темы
TOPICS = [
    "история",
    "наука",
    "география",
    "спорт",
    "кино и музыка",
    "технологии",
    "литература",
    "биология",
    "космос",
    "видеоигры",
]

DIFFICULTIES = ["easy", "medium", "hard"]


def _safe_parse_questions_json(content: str):
    if not content:
        return []

    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end <= start:
        return []

    try:
        payload = json.loads(content[start:end])
    except json.JSONDecodeError:
        return []

    questions = payload.get("questions") if isinstance(payload, dict) else None
    if questions is None and isinstance(payload, list):
        questions = payload

    if not isinstance(questions, list):
        return []

    valid = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        question_text = q.get("question")
        options = q.get("options")
        correct = q.get("correct")

        if not isinstance(question_text, str) or not question_text.strip():
            continue
        if not isinstance(options, list) or len(options) != 4:
            continue
        if not all(isinstance(o, str) and o.strip() for o in options):
            continue

        try:
            correct = int(correct)
        except Exception:
            continue
        if correct < 0 or correct > 3:
            continue

        valid.append({
            "question": question_text.strip(),
            "options": [o.strip() for o in options],
            "correct": correct,
        })

    return valid


def generate_questions(topic: str, difficulty: str, count: int = 35):
    """генерация вопросов через gemini"""

    if not GEMINI_API_KEY:
        print("[!] ошибка: нет GEMINI_API_KEY")
        return None

    prompt = f"""создай {count} вопросов для викторины на тему "{topic}" с уровнем сложности "{difficulty}" (на русском языке).

требования:
- вопросы должны быть интересными и разнообразными
- 4 варианта ответа на каждый вопрос
- только один правильный ответ
- не используй markdown, только plain text

ответь строго в формате json без пояснений:
{{
  "questions": [
    {{
      "question": "текст вопроса",
      "options": ["вариант 1", "вариант 2", "вариант 3", "вариант 4"],
      "correct": 0
    }}
  ]
}}

где correct - индекс (0-3) правильного ответа."""

    try:
        client = GeminiClient(
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            api_base_url=GEMINI_API_BASE_URL,
            timeout=120,
        )

        print(f"[*] отправка запроса для темы '{topic}' ({difficulty})...")
        resp = client.generate_content(
            prompt=prompt,
            use_search=False,
            temperature=0.7,
            max_tokens=3500,
            response_mime_type="application/json",
        )

        content = extract_text(resp)
        questions = _safe_parse_questions_json(content)
        if not questions:
            print("[!] не удалось распарсить вопросы из ответа")
            return None

        print(f"[+] получено {len(questions)} валидных вопросов")
        return questions

    except Exception as e:
        print(f"[!] ошибка: {e}")
        return None


def save_to_db(questions, topic: str, difficulty: str):
    """сохранение вопросов в базу данных"""

    from app import app, db, Question

    with app.app_context():
        for q in questions:
            try:
                question = Question(
                    topic=topic,
                    difficulty=difficulty,
                    question_text=q["question"],
                    option_1=q["options"][0],
                    option_2=q["options"][1],
                    option_3=q["options"][2],
                    option_4=q["options"][3],
                    correct_answer=q["correct"],
                )
                db.session.add(question)
            except Exception as e:
                print(f"[!] ошибка сохранения вопроса: {e}")
                continue

        db.session.commit()
        print("[+] вопросы сохранены в базу данных")


def main():
    """основная функция"""

    print("=" * 50)
    print("генератор вопросов для quizup")
    print("=" * 50)

    if not GEMINI_API_KEY:
        print("[!] укажите GEMINI_API_KEY в .env файле")
        return

    total_generated = 0

    for topic in TOPICS:
        for difficulty in DIFFICULTIES:
            print(f"\n[*] тема: {topic}, сложность: {difficulty}")

            questions = generate_questions(topic, difficulty, 35)

            if questions:
                save_to_db(questions, topic, difficulty)
                total_generated += len(questions)
            else:
                print("[!] не удалось сгенерировать вопросы")

            # небольшая пауза чтобы не перегружать api
            time.sleep(2)

    print("\n" + "=" * 50)
    print(f"всего сгенерировано: {total_generated} вопросов")
    print("=" * 50)


if __name__ == "__main__":
    main()

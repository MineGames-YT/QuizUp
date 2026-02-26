#!/usr/bin/env python3
"""скрипт для генерации вопросов через gemini

запуск:
    python generate_questions.py
"""

import time
from dotenv import load_dotenv

load_dotenv()


from app import app, save_questions_to_db, generate_questions_via_gemini


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


def main():
    print("=" * 50)
    print("генератор вопросов для QuizUp (gemini)")
    print("=" * 50)

    total_generated = 0

    with app.app_context():
        for topic in TOPICS:
            for difficulty in DIFFICULTIES:
                print(f"\n[*] тема: {topic}, сложность: {difficulty}")

                questions = generate_questions_via_gemini(topic, difficulty, 35)

                if questions:
                    save_questions_to_db(topic, questions, difficulty)
                    total_generated += len(questions)
                    print(f"[+] сохранено: {len(questions)}")
                else:
                    print("[!] не удалось сгенерировать вопросы")

                time.sleep(1)

    print("\n" + "=" * 50)
    print(f"всего сохранено: {total_generated} вопросов")
    print("=" * 50)


if __name__ == '__main__':
    main()

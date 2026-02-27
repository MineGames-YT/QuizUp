# QuizUp - командная викторина

приложение для проведения командных викторин в реальном времени.

## возможности

- две темы: светлая и тёмная
- два режима: команды (а vs б) и каждый сам за себя (ffa)
- три сложности: легко, средне, сложно
- бонус за скорость: чем быстрее ответил — тем больше очков
- 10 тем: история, наука, география, спорт, кино, технологии, литература, биология, космос, игры
- рейтинг игроков: система на основе elo
- админ-панель: пауза, пропуск, кик игроков
- экспорт результатов: в json

## запуск как десктопное приложение

### вариант 1: через python (простой)

```bash
# установка базовых зависимостей
pip install -r requirements.txt

# (опционально) зависимости для десктоп-оболочки
pip install -r requirements-desktop.txt

# запуск десктопной версии
python desktop.py
```

откроется окно приложения, сервер запустится автоматически.

### вариант 2: сборка exe (для windows)

```bash
# установка pyinstaller (лежит в requirements-desktop.txt)
pip install -r requirements-desktop.txt

# сборка
python build.py
```

готовый файл `QuizUp.exe` будет в папке `dist/`.

## запуск через браузер (классический способ)

```bash
python app.py
```

откройте http://localhost:5000 в браузере.

## настройка gemini

создайте файл `.env` и добавьте:

```bash
GEMINI_API_KEY=your_key_here
# стабильная "топовая" модель на февраль 2026:
GEMINI_MODEL=gemini-2.5-pro
# (опционально) если используете свой прокси/endpoint
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com
```

также можно настроить тайминги:

```bash
QUESTION_TIME_LIMIT=30
DISCONNECT_GRACE_SECONDS=12
NEXT_QUESTION_DELAY=2.0
```

## генерация вопросов в базу

```bash
python generate_questions.py
```

скрипт сгенерирует по 35 вопросов для каждой темы и сложности и сохранит их в sqlite.

## структура проекта

```
quizup/
├── app.py                 # основной сервер
├── desktop.py             # десктопная версия
├── build.py               # скрипт сборки exe
├── generate_questions.py  # генератор вопросов (gemini)
├── requirements.txt       # зависимости
├── requirements-desktop.txt # зависимости для десктоп/сборки
├── instance/
│   └── quizup.db          # sqlite база
├── static/
│   ├── css/style.css      # стили
│   └── sounds/            # звуки
└── templates/             # html шаблоны
```

## технологии

- flask + flask-socketio
- sqlalchemy (sqlite)
- flask-login (авторизация)
- pywebview (десктоп, опционально)
- pyinstaller (сборка exe, опционально)

## цветовая схема

- #104ba9 - основной синий
- #284a7e - тёмный синий
- #052d6e - глубокий синий
- #447bd4 - светлый синий
- #6a93d4 - акцентный синий

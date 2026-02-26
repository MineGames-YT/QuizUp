"""
quizbattle - командная викторина в реальном времени
полностью переписанная версия с бд, авторизацией и всеми фичами
"""

import os
import random
import string
import json
import base64
import time
import threading
import uuid
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from sqlalchemy import func
import requests
from dotenv import load_dotenv

# gigachat интеграция (ai генерация вопросов)
from gigachat_client import GigaChatClient, GigaChatAPIError
from gemini_client import GeminiClient, GeminiAPIError, extract_text as gemini_extract_text
from fallback_questions import FALLBACK_QUESTIONS

# загружаем переменные окружения из .env
load_dotenv()

# инициализация flask приложения
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///quizbattle.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# инициализация расширений flask
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==================== МОДЕЛИ БД ====================

class User(UserMixin, db.Model):
    """пользователь системы"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    avatar = db.Column(db.String(200), default='default.png')
    
    # статистика
    total_games = db.Column(db.Integer, default=0)
    total_wins = db.Column(db.Integer, default=0)
    total_points = db.Column(db.Integer, default=0)
    rating = db.Column(db.Integer, default=1000)  # elo-like рейтинг
    
    def __repr__(self):
        return f'<User {self.username}>'


class Question(db.Model):
    """вопрос для викторины"""
    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(50), nullable=False)
    difficulty = db.Column(db.String(20), default='medium')  # easy, medium, hard
    question_text = db.Column(db.Text, nullable=False)
    option_1 = db.Column(db.String(200), nullable=False)
    option_2 = db.Column(db.String(200), nullable=False)
    option_3 = db.Column(db.String(200), nullable=False)
    option_4 = db.Column(db.String(200), nullable=False)
    correct_answer = db.Column(db.Integer, nullable=False)  # 0-3
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    times_used = db.Column(db.Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question_text,
            'options': [self.option_1, self.option_2, self.option_3, self.option_4],
            'correct': self.correct_answer,
            'difficulty': self.difficulty
        }


class GameHistory(db.Model):
    """история игр"""
    id = db.Column(db.Integer, primary_key=True)
    pin = db.Column(db.String(6), nullable=False)
    topic = db.Column(db.String(50), nullable=False)
    mode = db.Column(db.String(20), default='teams')  # teams, ffa
    difficulty = db.Column(db.String(20), default='medium')
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)
    winner_team = db.Column(db.String(10))  # A, B, или user_id для ffa
    questions_count = db.Column(db.Integer, default=10)
    
    # связи
    creator = db.relationship('User', backref='created_games')
    players = db.relationship('PlayerStats', backref='game', lazy=True)


class PlayerStats(db.Model):
    """статистика игрока в конкретной игре"""
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey('game_history.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    guest_name = db.Column(db.String(50), nullable=True)  # для гостей
    team = db.Column(db.String(10))  # A, B или null для ffa
    score = db.Column(db.Integer, default=0)
    correct_answers = db.Column(db.Integer, default=0)
    wrong_answers = db.Column(db.Integer, default=0)
    avg_response_time = db.Column(db.Float, default=0)
    
    user = db.relationship('User', backref='game_stats')


# ==================== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================

# активные игры в памяти (для real-time)
active_games = {}
games_lock = threading.Lock()

# темы для выбора 
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

# конфигурация для API Кими
KIMI_API_KEY = os.environ.get('KIMI_API_KEY', '')
KIMI_API_URL = os.environ.get('KIMI_API_URL', 'https://api.moonshot.cn/v1/chat/completions')

# конфигурация gigachat (если хотите генерацию тем/вопросов через gigachat api)
GIGACHAT_AUTH_KEY = os.environ.get('GIGACHAT_AUTH_KEY', '').strip()
GIGACHAT_CLIENT_ID = os.environ.get('GIGACHAT_CLIENT_ID', '').strip()
GIGACHAT_CLIENT_SECRET = os.environ.get('GIGACHAT_CLIENT_SECRET', '').strip()

# если не передали готовую base64-строку, соберём её из id/secret
if not GIGACHAT_AUTH_KEY and GIGACHAT_CLIENT_ID and GIGACHAT_CLIENT_SECRET:
    raw = f"{GIGACHAT_CLIENT_ID}:{GIGACHAT_CLIENT_SECRET}".encode('utf-8')
    GIGACHAT_AUTH_KEY = base64.b64encode(raw).decode('utf-8')

GIGACHAT_SCOPE = os.environ.get('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS').strip()
GIGACHAT_MODEL = os.environ.get('GIGACHAT_MODEL', 'GigaChat').strip()

# в некоторых окружениях могут понадобиться сертификаты минцифры.
# если у вас сыпется ssl error - можно временно поставить false.
GIGACHAT_VERIFY_SSL_CERTS = os.environ.get('GIGACHAT_VERIFY_SSL_CERTS', 'true').lower() in ('1', 'true', 'yes', 'y')
# конфигурация gemini (google ai for developers)
# ключ берётся из google ai studio: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro").strip()
GEMINI_API_BASE_URL = os.environ.get("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com").strip()

# провайдер генерации вопросов по умолчанию (auto/gigachat/gemini/kimi/db)
DEFAULT_AI_PROVIDER = os.environ.get("DEFAULT_AI_PROVIDER", "auto").strip().lower()
DEFAULT_GEMINI_SEARCH = os.environ.get("DEFAULT_GEMINI_SEARCH", "false").lower() in ("1", "true", "yes", "y")


# игровой тайминг (по тз: 30 секунд на вопрос)
QUESTION_TIME_LIMIT = int(os.environ.get('QUESTION_TIME_LIMIT', '30'))

# если игрок просто перезагружает страницу/переходит из лобби в игру - даём пару секунд на реконнект
DISCONNECT_GRACE_SECONDS = int(os.environ.get('DISCONNECT_GRACE_SECONDS', '12'))

# пауза между вопросами, чтобы успела отыграть анимация/звук
NEXT_QUESTION_DELAY = float(os.environ.get('NEXT_QUESTION_DELAY', '2.0'))


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def generate_pin():
    """генерация уникального 6-значного пин-кода"""
    while True:
        pin = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        with games_lock:
            if pin not in active_games:
                return pin


def generate_questions_via_kimi(topic, difficulty, count=35):
    """генерация вопросов через API Кими"""
    if not KIMI_API_KEY:
        print("[!] нет API ключа")
        return None
    
    # составляем промпт для API
    prompt = f"""Создай {count} вопросов для викторины на тему "{topic}" с уровнем сложности "{difficulty}".

Требования:
- Вопросы должны быть разнообразными и интересными
- 4 варианта ответа на каждый вопрос
- Только один правильный ответ

Ответь СТРОГО в формате JSON:
{{
  "questions": [
    {{
      "question": "текст вопроса",
      "options": ["вариант 1", "вариант 2", "вариант 3", "вариант 4"],
      "correct": 0
    }}
  ]
}}

gде correct - индекс правильного ответа (0-3)."""

    try:
        headers = {
            'Authorization': f'Bearer {KIMI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'moonshot-v1-8k',
            'messages': [
                {'role': 'system', 'content': 'ты - генератор вопросов для викторины. отвечай только в формате json.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7
        }
        
        response = requests.post(KIMI_API_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # парсим json из ответа
            try:
                # ищем json в тексте
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    parsed = json.loads(json_str)
                    return parsed.get('questions', [])
            except json.JSONDecodeError:
                pass
                
    except Exception as e:
        print(f"ошибка генерации через kimi: {e}")
    
    return None



def _safe_parse_questions_json(content: str):
    """пытаемся вытащить список вопросов из ответа модели.

    модели иногда добавляют лишний текст, поэтому:
    - ищем json-объект в тексте
    - валидируем структуру вопросов
    """
    if not content:
        return []

    start = content.find('{')
    end = content.rfind('}') + 1

    if start == -1 or end <= start:
        return []

    try:
        payload = json.loads(content[start:end])
    except json.JSONDecodeError:
        return []

    questions = payload.get('questions') if isinstance(payload, dict) else None
    if questions is None and isinstance(payload, list):
        questions = payload

    if not isinstance(questions, list):
        return []

    valid = []
    for q in questions:
        if not isinstance(q, dict):
            continue

        question_text = q.get('question')
        options = q.get('options')
        correct = q.get('correct')

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
            'question': question_text.strip(),
            'options': [o.strip() for o in options],
            'correct': correct
        })

    return valid


def generate_questions_via_gigachat(topic, difficulty, count=20):
    """генерация вопросов через gigachat api"""
    if not GIGACHAT_AUTH_KEY:
        return None

    prompt = f"""создай {count} вопросов для викторины на тему \"{topic}\" с уровнем сложности \"{difficulty}\" (на русском языке).

требования:
- вопросы должны быть интересными и разнообразными
- 4 варианта ответа на каждый вопрос
- только один правильный ответ
- не используй markdown, только plain text

ответь строго в формате json без пояснений:
{{
  \"questions\": [
    {{
      \"question\": \"текст вопроса\",
      \"options\": [\"вариант 1\", \"вариант 2\", \"вариант 3\", \"вариант 4\"],
      \"correct\": 0
    }}
  ]
}}

где correct - индекс (0-3) правильного ответа."""

    try:
        client = GigaChatClient(
            credentials=GIGACHAT_AUTH_KEY,
            scope=GIGACHAT_SCOPE,
            model=GIGACHAT_MODEL,
            verify_ssl_certs=GIGACHAT_VERIFY_SSL_CERTS,
            timeout=90,
        )

        result = client.chat_completions(
            messages=[
                {'role': 'system', 'content': 'ты генератор вопросов для викторины. отвечай только json без markdown.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.7,
            max_tokens=3500,
        )

        content = result['choices'][0]['message']['content']
        parsed = _safe_parse_questions_json(content)
        return parsed if parsed else None

    except GigaChatAPIError as e:
        print(f"ошибка генерации через gigachat: {e} (status={e.status_code})")
        if e.details:
            print(e.details)
    except Exception as e:
        print(f"ошибка генерации через gigachat: {e}")

    return None



def generate_questions_via_gemini(topic, difficulty, count=20, *, use_search=False):
    """генерация вопросов через gemini api

    use_search=True включает google search grounding (если модель решит что нужно)
    """
    if not GEMINI_API_KEY:
        return None

    prompt = f"""создай {count} вопросов для викторины на тему \"{topic}\" с уровнем сложности \"{difficulty}\" (на русском языке).

требования:
- вопросы должны быть интересными и разнообразными
- 4 варианта ответа на каждый вопрос
- только один правильный ответ
- не используй markdown, только plain text

ответь строго в формате json без пояснений:
{{
  \"questions\": [
    {{
      \"question\": \"текст вопроса\",
      \"options\": [\"вариант 1\", \"вариант 2\", \"вариант 3\", \"вариант 4\"],
      \"correct\": 0
    }}
  ]
}}

gде correct - индекс (0-3) правильного ответа."""

    try:
        client = GeminiClient(
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            api_base_url=GEMINI_API_BASE_URL,
            timeout=90,
        )

        result = client.generate_content(
            prompt=prompt,
            use_search=bool(use_search),
            temperature=0.7,
            max_tokens=3500,
            response_mime_type="application/json",
        )

        content = gemini_extract_text(result)
        parsed = _safe_parse_questions_json(content)
        return parsed if parsed else None

    except GeminiAPIError as e:
        print(f"ошибка генерации через gemini: {e} (status={e.status_code})")
        if e.details:
            print(e.details)
    except Exception as e:
        print(f"ошибка генерации через gemini: {e}")

    return None
def save_questions_to_db(topic, questions, difficulty='medium'):
    """сохранение сгенерированных вопросов в бд"""
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
            print(f"ошибка сохранения вопроса: {e}")
            continue
    
    db.session.commit()


def get_random_questions(topic, count=10, difficulty=None):
    """получение случайных вопросов из бд"""
    query = Question.query.filter_by(topic=topic)
    
    if difficulty and difficulty != 'mixed':
        query = query.filter_by(difficulty=difficulty)
    
    questions = query.order_by(func.random()).limit(count).all()
    return [q.to_dict() for q in questions]


# ==================== КЛАСС ИГРЫ ====================

class GameSession:
    """класс управления игровой сессией"""

    def __init__(
        self,
        *,
        creator_id,
        creator_token: str | None,
        topic: str,
        mode: str = 'teams',
        difficulty: str = 'medium',
        questions_count: int = 5,
        has_password: bool = False,
        password: str | None = None,
        time_limit: int = QUESTION_TIME_LIMIT,
        bonus_enabled: bool = True,
        ai_provider: str = DEFAULT_AI_PROVIDER,
        gemini_search_enabled: bool = DEFAULT_GEMINI_SEARCH,
    ):
        self.pin = generate_pin()

        # автор/ведущий (для авторизованных пользователей) + токен ведущего (для гостей/сессий)
        self.creator_id = creator_id
        self.creator_token = creator_token

        # базовые настройки игры
        self.topic = topic
        self.mode = mode  # teams или ffa
        self.difficulty = difficulty

        # по тз в командном режиме количество выбирается "на команду"
        self.questions_per_team = int(questions_count) if self.mode == 'teams' else None
        self.questions_count = int(questions_count) * 2 if self.mode == 'teams' else int(questions_count)

        self.time_limit = int(time_limit)
        self.bonus_enabled = bool(bonus_enabled)

        # откуда добирать вопросы, если их не хватает в базе
        # auto: пытаемся по очереди доступные ключи (gigachat -> gemini -> kimi)
        self.ai_provider = (ai_provider or DEFAULT_AI_PROVIDER or 'auto').strip().lower()
        self.gemini_search_enabled = bool(gemini_search_enabled)

        self.has_password = bool(has_password)
        self.password = password

        self.status = 'waiting'
        self.created_at = time.time()

        # игроки
        # ключ - стабильный token игрока (лежит в localStorage), чтобы переживать перезагрузку страницы
        self.players: dict[str, dict] = {}  # token -> player_data
        self.sid_to_token: dict[str, str] = {}  # sid -> token (быстрый поиск на disconnect)

        self.teams: dict[str, list[str]] = {'A': [], 'B': []}  # team -> [player_token]

        # вопросы
        self.questions: list[dict] = []
        self.current_question_idx = 0

        # таймер и состояние раунда
        self.question_start_time: float | None = None
        self.question_timer_id: str | None = None
        self.question_timer: threading.Timer | None = None

        # для командного режима: один ответ на команду за вопрос
        self.round_answered = False
        self.current_team = 'A'

    # -------------------- игроки --------------------

    def _pick_team_for_new_player(self) -> str:
        """случайно-балансирующее распределение по командам"""
        a = len(self.teams['A'])
        b = len(self.teams['B'])

        if a < b:
            return 'A'
        if b < a:
            return 'B'

        return random.choice(['A', 'B'])

    def add_or_reconnect_player(self, *, sid: str, player_token: str, user_id=None, name: str | None = None):
        """добавить нового игрока или привязать новый sid к существующему token.

        возвращает (team, is_new)
        """
        if not player_token:
            # совсем без токена нельзя (нужно переживать переход лобби -> игра)
            player_token = uuid.uuid4().hex

        # реконнект
        if player_token in self.players:
            p = self.players[player_token]

            # если был таймер на удаление - отменяем
            t = p.get('disconnect_timer')
            if t:
                try:
                    t.cancel()
                except Exception:
                    pass
                p['disconnect_timer'] = None

            old_sid = p.get('sid')
            if old_sid and old_sid in self.sid_to_token:
                del self.sid_to_token[old_sid]

            p['sid'] = sid
            p['connected'] = True

            # имя можно обновлять (например, если игрок поменял его на форме)
            if name:
                p['name'] = name

            self.sid_to_token[sid] = player_token
            return p.get('team'), False

        # новый игрок
        team = None
        if self.mode == 'teams':
            team = self._pick_team_for_new_player()
            self.teams[team].append(player_token)

        self.players[player_token] = {
            'token': player_token,
            'sid': sid,
            'user_id': user_id,
            'name': name or 'игрок',
            'team': team,
            'score': 0,
            'correct': 0,
            'wrong': 0,
            'response_times': [],
            'answered_current': False,
            'connected': True,
            'disconnect_timer': None,
        }

        self.sid_to_token[sid] = player_token
        return team, True

    def mark_disconnected(self, sid: str):
        """отмечаем игрока как отключившегося и возвращаем token (если нашли)"""
        token = self.sid_to_token.get(sid)
        if not token:
            return None

        p = self.players.get(token)
        if not p:
            return None

        p['connected'] = False
        p['sid'] = None

        # sid уже не актуален
        del self.sid_to_token[sid]
        return token

    def remove_player(self, player_token: str):
        """полное удаление игрока из игры (после disconnect grace)"""
        if player_token not in self.players:
            return None

        p = self.players[player_token]
        team = p.get('team')
        if team and player_token in self.teams.get(team, []):
            self.teams[team].remove(player_token)

        # чистим sid mapping
        sid = p.get('sid')
        if sid and sid in self.sid_to_token:
            del self.sid_to_token[sid]

        # отменяем таймер на удаление (на всякий)
        t = p.get('disconnect_timer')
        if t:
            try:
                t.cancel()
            except Exception:
                pass

        name = p.get('name', 'игрок')
        del self.players[player_token]
        return name

    def get_player_token_by_sid(self, sid: str) -> str | None:
        return self.sid_to_token.get(sid)

    def get_player_by_sid(self, sid: str):
        token = self.get_player_token_by_sid(sid)
        if not token:
            return None
        return self.players.get(token)

    # -------------------- вопросы --------------------

    def load_questions(self):
        """загрузка вопросов из бд или генерация через ai"""
        need = self.questions_count
        questions = get_random_questions(self.topic, need, self.difficulty)

        # если в базе мало - пытаемся сгенерировать и докинуть
        if len(questions) < need:
            missing = need - len(questions)

            new_questions = None

            # выбираем источник генерации вопросов
            # важно: даже если выбран конкретный провайдер, при отсутствии ключа мы фоллбэкаем,
            # чтобы игра не умирала (иначе это будет боль в демо).
            provider = (self.ai_provider or 'auto').strip().lower()

            if provider == 'db':
                providers_order = []
            elif provider == 'auto':
                providers_order = ['gigachat', 'gemini', 'kimi']
            elif provider in ('gigachat', 'gemini', 'kimi'):
                providers_order = [provider, 'gigachat', 'gemini', 'kimi']
            else:
                providers_order = ['gigachat', 'gemini', 'kimi']

            # убираем дубли, сохраняя порядок
            uniq = []
            for p_name in providers_order:
                if p_name not in uniq:
                    uniq.append(p_name)

            for p_name in uniq:
                # берём с запасом, чтобы после фильтрации осталось нужное
                req_count = max(12, missing + 8)

                if p_name == 'gigachat' and GIGACHAT_AUTH_KEY:
                    new_questions = generate_questions_via_gigachat(self.topic, self.difficulty, req_count)

                elif p_name == 'gemini' and GEMINI_API_KEY:
                    new_questions = generate_questions_via_gemini(
                        self.topic,
                        self.difficulty,
                        req_count,
                        use_search=self.gemini_search_enabled,
                    )

                elif p_name == 'kimi' and KIMI_API_KEY:
                    new_questions = generate_questions_via_kimi(self.topic, self.difficulty, req_count)

                if new_questions:
                    break

            if new_questions:
                save_questions_to_db(self.topic, new_questions, self.difficulty)
                questions = get_random_questions(self.topic, need, self.difficulty)

        # если всё ещё не хватает - докидываем запасными вопросами, чтобы игра не умирала
        if len(questions) < need:
            missing = need - len(questions)
            pool = FALLBACK_QUESTIONS.copy()
            random.shuffle(pool)
            for q in pool[:missing]:
                questions.append({
                    'id': None,
                    'topic': self.topic,
                    'difficulty': self.difficulty,
                    'question': q['question'],
                    'options': q['options'],
                    'correct': q['correct'],
                })

        self.questions = questions[:need]

    def cancel_question_timer(self):
        """останавливаем таймер текущего вопроса"""
        self.question_timer_id = None
        self.question_start_time = None
        if self.question_timer:
            try:
                self.question_timer.cancel()
            except Exception:
                pass
        self.question_timer = None

    def start_question_timer(self, *, pin: str):
        """стартует таймер текущего вопроса (если он ещё не запущен)"""
        if self.question_start_time is not None and self.question_timer_id is not None:
            return

        self.question_start_time = time.time()
        self.question_timer_id = uuid.uuid4().hex

        timer_id = self.question_timer_id

        # запускаем отдельный таймер на истечение времени
        self.question_timer = threading.Timer(self.time_limit, lambda: time_up(pin, timer_id))
        self.question_timer.daemon = True
        self.question_timer.start()

    def get_current_question(self):
        """получение текущего вопроса (без вычисления очереди/таймера)"""
        if 0 <= self.current_question_idx < len(self.questions):
            q = self.questions[self.current_question_idx]
            return {
                'question': q['question'],
                'options': q['options'],
                'question_number': self.current_question_idx + 1,
                'total': len(self.questions),
                'current_team': self.current_team if self.mode == 'teams' else None,
            }
        return None

    def get_time_left(self) -> int:
        if not self.question_start_time:
            return self.time_limit
        elapsed = time.time() - self.question_start_time
        return max(0, int(round(self.time_limit - elapsed)))

    def check_answer(self, answer_idx: int) -> bool:
        """проверка ответа"""
        if 0 <= self.current_question_idx < len(self.questions):
            try:
                answer_idx = int(answer_idx)
            except Exception:
                return False
            return answer_idx == self.questions[self.current_question_idx]['correct']
        return False

    def calculate_score(self, *, is_correct: bool, response_time: float) -> int:
        """расчет очков.

        базово: 1 балл за правильный ответ (как в тз).
        бонус за скорость оставили опционально и очень мягким, чтобы не раздувать счёт.
        """
        if not is_correct:
            return 0

        score = 1

        if self.bonus_enabled:
            # +1 если уложился в первую треть времени (приятный бонус, но без перегиба)
            if response_time <= (self.time_limit / 3):
                score += 1

        return score

    def next_question(self) -> bool:
        """переход к следующему вопросу"""
        self.current_question_idx += 1

        # сбрасываем флаги ответов
        for p in self.players.values():
            p['answered_current'] = False

        # меняем команду
        if self.mode == 'teams':
            self.current_team = 'B' if self.current_team == 'A' else 'A'
            self.round_answered = False

        # таймер нового вопроса запустим при первом запросе get_question (или сразу ведущим)
        self.cancel_question_timer()

        return self.current_question_idx < len(self.questions)

    def get_leaderboard(self):
        """получение таблицы лидеров"""
        if self.mode == 'teams':
            score_a = sum(p['score'] for p in self.players.values() if p.get('team') == 'A')
            score_b = sum(p['score'] for p in self.players.values() if p.get('team') == 'B')
            return {'A': score_a, 'B': score_b}

        # ffa режим
        sorted_players = sorted(
            self.players.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        return [{'name': p['name'], 'score': p['score']} for p in sorted_players]

    def get_stats(self):
        """детальная статистика для результатов"""
        stats = []
        for p in self.players.values():
            avg_time = sum(p['response_times']) / len(p['response_times']) if p['response_times'] else 0
            stats.append({
                'name': p['name'],
                'team': p.get('team'),
                'score': p['score'],
                'correct': p['correct'],
                'wrong': p['wrong'],
                'avg_time': round(avg_time, 2),
            })
        return sorted(stats, key=lambda x: x['score'], reverse=True)


# ==================== РОУТЫ ====================

@app.route('/')
def index():
    """главная страница"""
    return render_template('index.html', topics=TOPICS)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """регистрация нового пользователя"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not username or not email or not password:
            flash('заполните все поля')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('такой username уже занят')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('такой email уже зарегистрирован')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=bcrypt.generate_password_hash(password).decode('utf-8')
        )
        db.session.add(user)
        db.session.commit()
        
        flash('регистрация успешна! теперь войдите')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """вход в систему"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        
        flash('неверный логин или пароль')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """выход из системы"""
    logout_user()
    return redirect(url_for('index'))


@app.route('/profile')
@login_required
def profile():
    """профиль пользователя"""
    # статистика игр
    stats = PlayerStats.query.filter_by(user_id=current_user.id).all()
    
    # позиция в рейтинге
    rank = User.query.filter(User.rating > current_user.rating).count() + 1
    
    return render_template('profile.html', stats=stats, rank=rank)


@app.route('/rating')
def rating():
    """таблица рейтинга - показываем только тех кто хоть раз играл"""
    # показываем только тех кто хотя бы раз играл
    users = User.query.filter(User.total_games > 0).order_by(User.rating.desc()).limit(100).all()
    return render_template('rating.html', users=users)


@app.route('/join', methods=['POST'])
def join():
    """присоединение к игре по POST"""
    pin = request.form.get('pin', '').upper().strip()
    guest_name = request.form.get('guest_name', '').strip()
    password = request.form.get('password', '')
    
    if not pin or not guest_name:
        flash('заполните все поля')
        return redirect(url_for('index'))
    
    # проверяем существование игры
    with games_lock:
        if pin not in active_games:
            flash('игра не найдена')
            return redirect(url_for('index'))
        
        game = active_games[pin]
        if game.status != 'waiting':
            flash('игра уже началась')
            return redirect(url_for('index'))
        
        if game.has_password and game.password != password:
            flash('неверный пароль')
            return redirect(url_for('index'))
    
    # сохраняем в сессию для websocket
    session['game_pin'] = pin
    session['guest_name'] = guest_name
    
    return redirect(url_for('lobby', pin=pin))


@app.route('/lobby')
def lobby():
    """страница лобби"""
    pin = request.args.get('pin', '').upper()
    if not pin:
        return redirect(url_for('index'))
    return render_template('lobby.html', pin=pin)


@app.route('/game')
def game():
    """игровая страница"""
    pin = request.args.get('pin', '').upper()
    if not pin:
        return redirect(url_for('index'))
    return render_template('game.html', pin=pin)


# ==================== SOCKET.IO ====================

def end_game(pin: str):
    """завершение игры и сохранение результатов в бд"""

    # снимем слепок игры под локом (дальше будем писать в базу без гонок)
    with games_lock:
        game = active_games.get(pin)
        if not game:
            return

        # защита от двойного завершения (например, таймер + админ)
        if game.status == 'finished':
            return

        game.cancel_question_timer()
        game.status = 'finished'

        mode = game.mode
        topic = game.topic
        leaderboard = game.get_leaderboard()
        stats = game.get_stats()

        players_snapshot = []
        for p in game.players.values():
            players_snapshot.append({
                'user_id': p.get('user_id'),
                'name': p.get('name'),
                'team': p.get('team'),
                'score': int(p.get('score', 0)),
                'correct': int(p.get('correct', 0)),
                'wrong': int(p.get('wrong', 0)),
                'avg_time': (sum(p.get('response_times', [])) / len(p.get('response_times', []))) if p.get('response_times') else 0,
            })

        # определяем победителя
        winner = None
        if mode == 'teams':
            a = leaderboard.get('A', 0)
            b = leaderboard.get('B', 0)
            if a > b:
                winner = 'A'
            elif b > a:
                winner = 'B'
            else:
                winner = 'draw'
        else:
            if stats:
                winner = stats[0]['name']

        # активную игру убираем (результаты уже улетят всем через socket)
        del active_games[pin]

    # db операции - только в app context (важно для таймеров)
    with app.app_context():
        history = GameHistory.query.filter_by(pin=pin).order_by(GameHistory.created_at.desc()).first()
        if history and history.ended_at is None:
            history.ended_at = datetime.utcnow()
            history.winner_team = winner

            # сохраняем статистику игроков
            for p in players_snapshot:
                ps = PlayerStats(
                    game_id=history.id,
                    user_id=p['user_id'],
                    guest_name=None if p['user_id'] else p['name'],
                    team=p['team'],
                    score=p['score'],
                    correct_answers=p['correct'],
                    wrong_answers=p['wrong'],
                    avg_response_time=p['avg_time'],
                )
                db.session.add(ps)

            # обновляем общий рейтинг пользователей
            for p in players_snapshot:
                if not p['user_id']:
                    continue
                user = User.query.get(p['user_id'])
                if not user:
                    continue

                user.total_games += 1
                user.total_points += p['score']

                if winner:
                    if mode == 'teams' and p['team'] == winner:
                        user.wins += 1
                    if mode == 'ffa' and p['name'] == winner:
                        user.wins += 1

                # простая формула рейтинга (можно усложнять, но для прототипа ок)
                user.rating = round((user.wins / user.total_games) * 100, 1) if user.total_games else 0

            db.session.commit()

    socketio.emit('game_finished', {
        'leaderboard': leaderboard,
        'stats': stats,
        'topic': topic,
        'mode': mode,
        'winner': winner,
    }, room=pin)


def time_up(pin: str, timer_id: str):
    """срабатывает, когда время на вопрос закончилось"""
    with games_lock:
        game = active_games.get(pin)
        if not game or game.status != 'playing':
            return

        # если таймер уже не актуален (например, вопрос закрыли ответом) - просто игнор
        if game.question_timer_id != timer_id:
            return

        # закрываем текущий вопрос
        game.cancel_question_timer()

        # в ffa можно начислить промах тем, кто не успел
        if game.mode == 'ffa':
            for p in game.players.values():
                if not p.get('answered_current'):
                    p['wrong'] += 1
                    p['answered_current'] = True

    socketio.emit('time_up', room=pin)

    # двигаемся дальше через небольшую паузу
    _schedule_next_question(pin)


@socketio.on('connect')
def handle_connect():
    """подключение клиента"""
    # просто коннект, ничего интересного
    pass


@socketio.on('disconnect')
def handle_disconnect():
    """отключение клиента

    важный момент:
    при переходе лобби -> игра страница перезагружается, сокет пересоздаётся и мы ловим disconnect.
    поэтому удаляем игрока не мгновенно, а с небольшим grace-периодом.
    """
    pin_found = None
    token = None

    with games_lock:
        for pin, game in list(active_games.items()):
            token = game.get_player_token_by_sid(request.sid)
            if not token:
                continue

            p = game.players.get(token)
            if not p:
                continue

            pin_found = pin

            # помечаем как оффлайн
            game.mark_disconnected(request.sid)

            # планируем удаление, если игрок не вернётся
            def _cleanup_disconnect(pin=pin, player_token=token):
                removed_name = None
                players_payload = []

                with games_lock:
                    g = active_games.get(pin)
                    if not g:
                        return

                    p2 = g.players.get(player_token)
                    if not p2:
                        return

                    # успел переподключиться
                    if p2.get('connected'):
                        return

                    removed_name = g.remove_player(player_token)
                    players_payload = get_players_list(g)

                    # если никого не осталось - закрываем комнату
                    if len(g.players) == 0:
                        del active_games[pin]

                if removed_name:
                    socketio.emit('player_left', {
                        'name': removed_name,
                        'players': players_payload
                    }, room=pin)

            timer = threading.Timer(DISCONNECT_GRACE_SECONDS, _cleanup_disconnect)
            timer.daemon = True
            p['disconnect_timer'] = timer
            timer.start()

            break  # sid может быть только в одной игре

    # не шлём player_left сразу: игрок может просто перезагрузить страницу



def get_players_list(game):
    """список игроков для фронта (без внутренних токенов)"""
    result = []
    for p in game.players.values():
        result.append({
            'name': p['name'],
            'team': p.get('team'),
            'score': p.get('score', 0),
            'connected': bool(p.get('connected', True)),
        })
    return result


@socketio.on('create_game')
def handle_create_game(data):
    """создание новой игры"""
    topic = (data.get('topic') or '').strip()
    mode = (data.get('mode') or 'teams').strip()
    difficulty = (data.get('difficulty') or 'medium').strip()
    questions_count = int(data.get('questions_count', 5))
    has_password = bool(data.get('has_password', False))
    password = (data.get('password') or '').strip() if has_password else None
    bonus_enabled = bool(data.get('bonus_enabled', True))

    ai_provider = (data.get('ai_provider') or DEFAULT_AI_PROVIDER or 'auto').strip().lower()
    gemini_search_enabled = bool(data.get('gemini_search_enabled', DEFAULT_GEMINI_SEARCH))

    if ai_provider not in ('auto', 'gigachat', 'gemini', 'kimi', 'db'):
        ai_provider = DEFAULT_AI_PROVIDER if DEFAULT_AI_PROVIDER in ('auto', 'gigachat', 'gemini', 'kimi', 'db') else 'auto'

    # стабильный токен игрока (лежит в localStorage). используем как "ключ ведущего"
    creator_token = (data.get('player_token') or '').strip() or uuid.uuid4().hex

    if not topic:
        emit('error_message', {'message': 'укажи тему для игры'})
        return

    # в базе поле topic ограничено 50 символами, режем аккуратно
    topic = topic[:50].lower()

    # валидируем минимально, чтобы не ловить странные состояния
    if mode not in ('teams', 'ffa'):
        mode = 'teams'
    if difficulty not in ('easy', 'medium', 'hard'):
        difficulty = 'medium'

    # по тз: 5/6/7 вопросов на команду
    if mode == 'teams' and questions_count not in (5, 6, 7):
        questions_count = 5

    game = GameSession(
        creator_id=current_user.id if current_user.is_authenticated else None,
        creator_token=creator_token,
        topic=topic,
        mode=mode,
        difficulty=difficulty,
        questions_count=questions_count,
        has_password=has_password,
        password=password,
        time_limit=QUESTION_TIME_LIMIT,
        bonus_enabled=bonus_enabled,
        ai_provider=ai_provider,
        gemini_search_enabled=gemini_search_enabled,
    )

    with games_lock:
        active_games[game.pin] = game

    # на странице index это соединение почти сразу умрёт из-за redirect, но пусть будет
    join_room(game.pin)

    emit('game_created', {
        'pin': game.pin,
        'topic': game.topic,
        'mode': game.mode,
        'difficulty': game.difficulty,
        'questions_count': game.questions_count,
        'questions_per_team': game.questions_per_team,
        'time_limit': game.time_limit,
        'ai_provider': game.ai_provider,
        'gemini_search_enabled': bool(game.gemini_search_enabled),
    })


def _normalize_player_name(raw_name: str, player_token: str):
    name = (raw_name or '').strip()
    if name:
        return name[:20]
    # чтобы в лобби не было 5 одинаковых "игрок"
    return f"игрок-{player_token[:4]}"


def _is_creator(game: GameSession, *, actor_token: str | None):
    # авторизованный ведущий
    if current_user.is_authenticated and game.creator_id and current_user.id == game.creator_id:
        return True
    # гостевой ведущий
    if actor_token and game.creator_token and actor_token == game.creator_token:
        return True
    return False


def _schedule_next_question(pin: str):
    """переходим на следующий вопрос через небольшую паузу"""

    def _advance():
        with games_lock:
            game = active_games.get(pin)
            if not game:
                return
            if game.status != 'playing':
                return

            has_next = game.next_question()

        if not has_next:
            end_game(pin)
            return

        socketio.emit('next_question_ready', room=pin)

    t = threading.Timer(NEXT_QUESTION_DELAY, _advance)
    t.daemon = True
    t.start()


@socketio.on('join_game')
def handle_join_game(data):
    """присоединение к игре (и реконнект)"""
    pin = (data.get('pin') or '').upper().strip()
    player_token = (data.get('player_token') or '').strip()
    guest_name = data.get('guest_name', '')
    password = data.get('password')

    if not pin:
        emit('error_message', {'message': 'неверный pin'})
        return

    if not player_token:
        # без токена не сможем отличать реконнект от нового игрока
        player_token = uuid.uuid4().hex

    with games_lock:
        game = active_games.get(pin)
        if not game:
            emit('error_message', {'message': 'игра не найдена'})
            return

        # проверка пароля нужна только на вход в лобби (waiting). при реконнекте в игре не трогаем.
        if game.status == 'waiting' and game.has_password:
            if password != game.password:
                emit('error_message', {'message': 'неверный пароль'})
                return

        # если игра идёт - впускаем только реконнекты (по token)
        if game.status in ('playing', 'paused'):
            if player_token not in game.players:
                emit('error_message', {'message': 'игра уже началась'})
                return

        if game.status == 'finished':
            emit('error_message', {'message': 'игра уже закончилась'})
            return

        # нормализуем имя (для гостей можно из localStorage, для логина - username)
        if current_user.is_authenticated:
            final_name = current_user.username
        else:
            final_name = _normalize_player_name(guest_name, player_token)

        team, is_new = game.add_or_reconnect_player(
            sid=request.sid,
            player_token=player_token,
            user_id=current_user.id if current_user.is_authenticated else None,
            name=final_name,
        )

        join_room(pin)

        is_creator = _is_creator(game, actor_token=player_token)

        joined_payload = {
            'pin': pin,
            'name': final_name,
            'team': team,
            'mode': game.mode,
            'topic': game.topic,
            'difficulty': game.difficulty,
            'questions_count': game.questions_count,
            'questions_per_team': game.questions_per_team,
            'time_limit': game.time_limit,
            'status': game.status,
            'leaderboard': game.get_leaderboard(),
            'is_creator': is_creator,
            'players': get_players_list(game),
        }

    emit('joined', joined_payload, to=request.sid)

    # обновляем список игроков всем в комнате
    socketio.emit('player_joined', {
        'name': final_name,
        'players': joined_payload['players']
    }, room=pin)


@socketio.on('start_game')
def handle_start_game(data):
    """старт игры"""
    pin = (data.get('pin') or '').upper().strip()
    actor_token = (data.get('player_token') or data.get('creator_token') or '').strip()

    with games_lock:
        game = active_games.get(pin)
        if not game:
            return

        if game.status != 'waiting':
            return

        if not _is_creator(game, actor_token=actor_token):
            emit('error_message', {'message': 'только ведущий может начать игру'})
            return

        if len(game.players) < 2:
            emit('error_message', {'message': 'нужно минимум 2 игрока'})
            return

        # подгружаем вопросы (бд -> gigachat/kimi -> fallback)
        game.load_questions()
        if not game.questions or len(game.questions) < game.questions_count:
            emit('error_message', {'message': 'не удалось загрузить вопросы'})
            return

        game.status = 'playing'
        game.current_question_idx = 0
        game.current_team = 'A'
        game.round_answered = False
        game.cancel_question_timer()

        # сохраняем в историю
        history = GameHistory(
            pin=game.pin,
            created_by=game.creator_id,
            topic=game.topic,
            mode=game.mode,
            difficulty=game.difficulty,
            questions_count=game.questions_count,
        )
        db.session.add(history)
        db.session.commit()

    socketio.emit('game_started', {
        'pin': pin,
        'topic': game.topic,
        'mode': game.mode,
        'difficulty': game.difficulty,
        'questions_count': game.questions_count,
        'questions_per_team': game.questions_per_team,
        'time_limit': game.time_limit,
    }, room=pin)


@socketio.on('get_question')
def handle_get_question(data):
    """получение текущего вопроса (в т.ч. для реконнекта)"""
    pin = (data.get('pin') or '').upper().strip()

    with games_lock:
        game = active_games.get(pin)
        if not game:
            return
        if game.status != 'playing':
            return

        player = game.get_player_by_sid(request.sid)
        if not player:
            emit('error_message', {'message': 'ты не в этой игре'})
            return

        # таймер запускаем один раз на вопрос
        game.start_question_timer(pin=pin)

        q = game.get_current_question()
        if not q:
            return

        player_team = player.get('team')
        is_your_turn = (game.mode == 'ffa') or (player_team == game.current_team)

        payload = {
            **q,
            'is_your_turn': is_your_turn,
            'time_left': game.get_time_left(),
            'leaderboard': game.get_leaderboard(),
        }

    emit('question', payload, to=request.sid)


@socketio.on('submit_answer')
def handle_submit_answer(data):
    """ответ игрока"""
    pin = (data.get('pin') or '').upper().strip()
    answer = data.get('answer')

    try:
        answer = int(answer)
    except Exception:
        return

    with games_lock:
        game = active_games.get(pin)
        if not game or game.status != 'playing':
            return

        player = game.get_player_by_sid(request.sid)
        if not player:
            return

        # если таймер ещё не стартовал - считаем, что вопрос не получен
        if not game.question_start_time:
            return

        # режим команд: отвечает только текущая команда, и только один ответ на вопрос
        if game.mode == 'teams':
            if player.get('team') != game.current_team:
                return
            if game.round_answered:
                return
        else:
            # ffa: каждый отвечает один раз
            if player.get('answered_current'):
                return

        is_correct = game.check_answer(answer)
        response_time = time.time() - game.question_start_time
        points = game.calculate_score(is_correct=is_correct, response_time=response_time)

        # обновляем статистику игрока
        player['answered_current'] = True
        player['response_times'].append(response_time)

        if is_correct:
            player['score'] += points
            player['correct'] += 1
        else:
            player['wrong'] += 1

        answered_by = player['name']

        # в командном режиме сразу закрываем раунд
        if game.mode == 'teams':
            game.round_answered = True

        # если раунд завершён раньше времени - глушим таймер
        if game.mode == 'teams':
            game.cancel_question_timer()

        leaderboard = game.get_leaderboard()
        current_team = game.current_team

        # в ffa проверим, все ли уже ответили
        all_answered = False
        if game.mode == 'ffa':
            all_answered = all(p.get('answered_current') for p in game.players.values())
            if all_answered:
                game.cancel_question_timer()

    # ответ показываем только тому, кто нажал
    emit('answer_result', {
        'correct': is_correct,
        'answer': answer,
        'score': points,
    }, to=request.sid)

    socketio.emit('score_update', {
        'leaderboard': leaderboard,
        'answered_by': answered_by,
        'is_correct': is_correct,
    }, room=pin)

    # если ответила команда - блокируем оставшихся в этой команде (чтобы не кликали дальше)
    if game.mode == 'teams':
        socketio.emit('round_locked', {
            'team': current_team,
            'answered_by': answered_by,
            'correct': is_correct,
        }, room=pin)
        _schedule_next_question(pin)
        return

    # ffa: ждём всех игроков, потом следующий вопрос
    if all_answered:
        _schedule_next_question(pin)


@socketio.on('admin_pause')
def handle_pause(data):
    """пауза/продолжить (только ведущий)"""
    pin = (data.get('pin') or '').upper().strip()
    actor_token = (data.get('player_token') or '').strip()

    with games_lock:
        game = active_games.get(pin)
        if not game:
            return

        if not _is_creator(game, actor_token=actor_token):
            return

        if game.status == 'playing':
            game.status = 'paused'
            game.cancel_question_timer()
            socketio.emit('game_paused', room=pin)
            return

        if game.status == 'paused':
            game.status = 'playing'
            # таймер стартанёт на следующем get_question (клиенты сами дёрнут)
            socketio.emit('game_resumed', room=pin)
            return


@socketio.on('admin_skip')
def handle_skip(data):
    """пропуск вопроса (только ведущий)"""
    pin = (data.get('pin') or '').upper().strip()
    actor_token = (data.get('player_token') or '').strip()

    with games_lock:
        game = active_games.get(pin)
        if not game:
            return
        if game.status != 'playing':
            return
        if not _is_creator(game, actor_token=actor_token):
            return

        game.cancel_question_timer()
        has_next = game.next_question()

    if not has_next:
        end_game(pin)
        return

    socketio.emit('next_question_ready', room=pin)


@socketio.on('admin_end')
def handle_admin_end(data):
    """досрочное завершение игры (только ведущий)"""
    pin = (data.get('pin') or '').upper().strip()
    actor_token = (data.get('player_token') or '').strip()

    with games_lock:
        game = active_games.get(pin)
        if not game:
            return
        if not _is_creator(game, actor_token=actor_token):
            return

    end_game(pin)


@socketio.on('admin_kick')
def handle_kick(data):
    """исключение игрока из игры (только ведущий)"""
    pin = (data.get('pin') or '').upper().strip()
    actor_token = (data.get('player_token') or '').strip()
    target_name = (data.get('target_name') or '').strip()
    target_token = (data.get('target_token') or '').strip()

    removed_name = None

    with games_lock:
        game = active_games.get(pin)
        if not game:
            return
        if not _is_creator(game, actor_token=actor_token):
            return

        # если пришёл точный token - отлично
        if target_token and target_token in game.players:
            removed_name = game.remove_player(target_token)
        else:
            # иначе пытаемся по имени (не идеально, но лучше чем ничего)
            if target_name:
                for tok, p in list(game.players.items()):
                    if p.get('name') == target_name:
                        removed_name = game.remove_player(tok)
                        break

    if removed_name:
        socketio.emit('player_kicked', {'name': removed_name}, room=pin)
        socketio.emit('player_left', {
            'name': removed_name,
            'players': get_players_list(game)
        }, room=pin)



# ==================== API РОУТЫ ====================

@app.route('/api/game/<pin>/stats')
def get_game_stats(pin):
    """получение статистики игры"""
    with games_lock:
        if pin in active_games:
            game = active_games[pin]
            return jsonify(game.get_stats())
    
    # если игра уже в бд
    history = GameHistory.query.filter_by(pin=pin).order_by(GameHistory.created_at.desc()).first()
    if history:
        stats = PlayerStats.query.filter_by(game_id=history.id).all()
        return jsonify([{
            'name': s.guest_name or (s.user.username if s.user else 'игрок'),
            'team': s.team,
            'score': s.score,
            'correct': s.correct_answers,
            'wrong': s.wrong_answers,
            'avg_time': s.avg_response_time
        } for s in stats])
    
    return jsonify({'error': 'игра не найдена'}), 404


@app.route('/api/game/<pin>/export')
def export_game_results(pin):
    """экспорт результатов в json"""
    history = GameHistory.query.filter_by(pin=pin).order_by(GameHistory.created_at.desc()).first()
    if not history:
        return jsonify({'error': 'игра не найдена'}), 404
    
    stats = PlayerStats.query.filter_by(game_id=history.id).all()
    
    data = {
        'pin': pin,
        'topic': history.topic,
        'mode': history.mode,
        'difficulty': history.difficulty,
        'created_at': history.created_at.isoformat(),
        'ended_at': history.ended_at.isoformat() if history.ended_at else None,
        'winner': history.winner_team,
        'players': [{
            'name': s.guest_name or (s.user.username if s.user else 'игрок'),
            'team': s.team,
            'score': s.score,
            'correct': s.correct_answers,
            'wrong': s.wrong_answers
        } for s in stats]
    }
    
    # сохраняем во временный файл
    filename = f'game_{pin}_{int(time.time())}.json'
    filepath = os.path.join('/tmp', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return send_file(filepath, as_attachment=True, download_name=filename)


# ==================== ИНИЦИАЛИЗАЦИЯ ====================

def init_db():
    """инициализация базы данных"""
    with app.app_context():
        db.create_all()
        print("база данных создана")


if __name__ == '__main__':
    init_db()
    print("=" * 50)
    print("quizbattle сервер запущен")
    print("=" * 50)
    print("открой http://localhost:5000 в браузере")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

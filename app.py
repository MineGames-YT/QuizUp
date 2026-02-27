"""
quizbattle - командная викторина в реальном времени
"""

import os
import random
import string
import json
import time
import threading
import uuid
import tempfile
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_socketio import SocketIO, emit, join_room
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from sqlalchemy import func
from dotenv import load_dotenv

# Gemini интеграция
from gemini_client import GeminiClient, GeminiAPIError, extract_text as gemini_extract_text
from fallback_questions import FALLBACK_QUESTIONS

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///quizbattle.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==================== МОДЕЛИ БД ====================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    avatar = db.Column(db.String(200), default='default.png')
    total_games = db.Column(db.Integer, default=0)
    total_wins = db.Column(db.Integer, default=0)
    total_points = db.Column(db.Integer, default=0)
    rating = db.Column(db.Integer, default=1000)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(50), nullable=False)
    difficulty = db.Column(db.String(20), default='medium')
    question_text = db.Column(db.Text, nullable=False)
    option_1 = db.Column(db.String(200), nullable=False)
    option_2 = db.Column(db.String(200), nullable=False)
    option_3 = db.Column(db.String(200), nullable=False)
    option_4 = db.Column(db.String(200), nullable=False)
    correct_answer = db.Column(db.Integer, nullable=False)
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
    id = db.Column(db.Integer, primary_key=True)
    pin = db.Column(db.String(6), nullable=False)
    topic = db.Column(db.String(50), nullable=False)
    mode = db.Column(db.String(20), default='teams')
    difficulty = db.Column(db.String(20), default='medium')
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)
    winner_team = db.Column(db.String(10))
    questions_count = db.Column(db.Integer, default=10)
    creator = db.relationship('User', backref='created_games')
    players = db.relationship('PlayerStats', backref='game', lazy=True)

class PlayerStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey('game_history.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    guest_name = db.Column(db.String(50), nullable=True)
    team = db.Column(db.String(10))
    score = db.Column(db.Integer, default=0)
    correct_answers = db.Column(db.Integer, default=0)
    wrong_answers = db.Column(db.Integer, default=0)
    avg_response_time = db.Column(db.Float, default=0)
    user = db.relationship('User', backref='game_stats')

# ==================== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================

active_games = {}
games_lock = threading.Lock()

TOPICS = [
    'история', 'наука', 'география', 'спорт', 'кино и музыка',
    'технологии', 'литература', 'биология', 'космос', 'видеоигры'
]

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro").strip()
GEMINI_API_BASE_URL = os.environ.get("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com").strip()

QUESTION_TIME_LIMIT = int(os.environ.get('QUESTION_TIME_LIMIT', '30'))
DISCONNECT_GRACE_SECONDS = int(os.environ.get('DISCONNECT_GRACE_SECONDS', '12'))
NEXT_QUESTION_DELAY = float(os.environ.get('NEXT_QUESTION_DELAY', '2.0'))

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def generate_pin():
    while True:
        pin = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        with games_lock:
            if pin not in active_games:
                return pin

def _safe_parse_questions_json(content: str):
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

def generate_questions_via_gemini(topic, difficulty, count=20):
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

где correct - индекс (0-3) правильного ответа."""
    try:
        client = GeminiClient(
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            api_base_url=GEMINI_API_BASE_URL,
            timeout=90,
        )
        result = client.generate_content(
            prompt=prompt,
            use_search=False,
            temperature=0.7,
            max_tokens=3500,
            response_mime_type="application/json",
        )
        content = gemini_extract_text(result)
        parsed = _safe_parse_questions_json(content)
        return parsed if parsed else None
    except Exception as e:
        print(f"ошибка генерации через gemini: {e}")
        return None

def save_questions_to_db(topic, questions, difficulty='medium'):
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
    query = Question.query.filter_by(topic=topic)
    if difficulty and difficulty != 'mixed':
        query = query.filter_by(difficulty=difficulty)
    questions = query.order_by(func.random()).limit(count).all()
    return [q.to_dict() for q in questions]

# ==================== КЛАСС ИГРЫ ====================

class GameSession:
    def __init__(
        self,
        *,
        creator_id,
        creator_token: str | None,
        topic: str,
        mode: str = 'teams',
        difficulty: str = 'medium',
        questions_count: int = 5,
        time_limit: int = QUESTION_TIME_LIMIT,
        bonus_enabled: bool = True,
    ):
        self.pin = generate_pin()
        self.creator_id = creator_id
        self.creator_token = creator_token
        self.topic = topic
        self.mode = mode
        self.difficulty = difficulty
        self.questions_per_team = int(questions_count) if self.mode == 'teams' else None
        self.questions_count = int(questions_count) * 2 if self.mode == 'teams' else int(questions_count)
        self.time_limit = int(time_limit)
        self.bonus_enabled = bool(bonus_enabled)

        self.status = 'waiting'
        self.created_at = time.time()

        self.players: dict[str, dict] = {}
        self.sid_to_token: dict[str, str] = {}
        self.teams: dict[str, list[str]] = {'A': [], 'B': []}

        self.questions: list[dict] = []
        self.current_question_idx = 0

        self.question_start_time: float | None = None
        self.question_timer_id: str | None = None
        self.question_timer: threading.Timer | None = None
        self.round_answered = False
        self.current_team = 'A'

    def _pick_team_for_new_player(self) -> str:
        a = len(self.teams['A'])
        b = len(self.teams['B'])
        if a < b:
            return 'A'
        if b < a:
            return 'B'
        return random.choice(['A', 'B'])

    def add_or_reconnect_player(self, *, sid: str, player_token: str, user_id=None, name: str | None = None):
        if not player_token:
            player_token = uuid.uuid4().hex
        if player_token in self.players:
            p = self.players[player_token]
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
            if name:
                p['name'] = name
            self.sid_to_token[sid] = player_token
            return p.get('team'), False

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
        token = self.sid_to_token.get(sid)
        if not token:
            return None
        p = self.players.get(token)
        if not p:
            return None
        p['connected'] = False
        p['sid'] = None
        del self.sid_to_token[sid]
        return token

    def remove_player(self, player_token: str):
        if player_token not in self.players:
            return None
        p = self.players[player_token]
        team = p.get('team')
        if team and player_token in self.teams.get(team, []):
            self.teams[team].remove(player_token)
        sid = p.get('sid')
        if sid and sid in self.sid_to_token:
            del self.sid_to_token[sid]
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
        return self.players.get(token)

    def load_questions(self):
        need = self.questions_count
        questions = get_random_questions(self.topic, need, self.difficulty)
        if len(questions) < need and GEMINI_API_KEY:
            missing = need - len(questions)
            req_count = max(12, missing + 8)
            new_questions = generate_questions_via_gemini(self.topic, self.difficulty, req_count)
            if new_questions:
                save_questions_to_db(self.topic, new_questions, self.difficulty)
                questions = get_random_questions(self.topic, need, self.difficulty)
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
        self.question_timer_id = None
        self.question_start_time = None
        if self.question_timer:
            try:
                self.question_timer.cancel()
            except Exception:
                pass
        self.question_timer = None

    def start_question_timer(self, *, pin: str):
        if self.question_start_time is not None and self.question_timer_id is not None:
            return
        self.question_start_time = time.time()
        self.question_timer_id = uuid.uuid4().hex
        timer_id = self.question_timer_id
        self.question_timer = threading.Timer(self.time_limit, lambda: time_up(pin, timer_id))
        self.question_timer.daemon = True
        self.question_timer.start()

    def get_current_question(self):
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
        if 0 <= self.current_question_idx < len(self.questions):
            try:
                answer_idx = int(answer_idx)
            except Exception:
                return False
            return answer_idx == self.questions[self.current_question_idx]['correct']
        return False

    def calculate_score(self, *, is_correct: bool, response_time: float) -> int:
        if not is_correct:
            return 0
        score = 1
        if self.bonus_enabled and response_time <= (self.time_limit / 3):
            score += 1
        return score

    def next_question(self) -> bool:
        self.current_question_idx += 1
        for p in self.players.values():
            p['answered_current'] = False
        if self.mode == 'teams':
            self.current_team = 'B' if self.current_team == 'A' else 'A'
            self.round_answered = False
        self.cancel_question_timer()
        return self.current_question_idx < len(self.questions)

    def get_leaderboard(self):
        if self.mode == 'teams':
            score_a = sum(p['score'] for p in self.players.values() if p.get('team') == 'A')
            score_b = sum(p['score'] for p in self.players.values() if p.get('team') == 'B')
            return {'A': score_a, 'B': score_b}
        sorted_players = sorted(
            self.players.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        return [{'name': p['name'], 'score': p['score']} for p in sorted_players]

    def get_stats(self):
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
    return render_template('index.html', topics=TOPICS)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('заполните все поля')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('такой username уже занят')
            return redirect(url_for('register'))
        user = User(
            username=username,
            password_hash=bcrypt.generate_password_hash(password).decode('utf-8')
        )
        db.session.add(user)
        db.session.commit()
        flash('регистрация успешна! теперь войдите')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
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
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    stats = PlayerStats.query.filter_by(user_id=current_user.id).all()
    rank = User.query.filter(User.rating > current_user.rating).count() + 1
    return render_template('profile.html', stats=stats, rank=rank)

@app.route('/rating')
def rating():
    users = User.query.filter(User.total_games > 0).order_by(User.rating.desc()).limit(100).all()
    return render_template('rating.html', users=users)

@app.route('/join', methods=['POST'])
def join():
    pin = request.form.get('pin', '').upper().strip()
    guest_name = request.form.get('guest_name', '').strip()
    if not pin or not guest_name:
        flash('заполните все поля')
        return redirect(url_for('index'))
    with games_lock:
        if pin not in active_games:
            flash('игра не найдена')
            return redirect(url_for('index'))
        game = active_games[pin]
        if game.status != 'waiting':
            flash('игра уже началась')
            return redirect(url_for('index'))
    session['game_pin'] = pin
    session['guest_name'] = guest_name
    return redirect(url_for('lobby', pin=pin))

@app.route('/lobby')
def lobby():
    pin = request.args.get('pin', '').upper()
    if not pin:
        return redirect(url_for('index'))
    return render_template('lobby.html', pin=pin)

@app.route('/game')
def game():
    pin = request.args.get('pin', '').upper()
    if not pin:
        return redirect(url_for('index'))
    return render_template('game.html', pin=pin)

# ==================== SOCKET.IO ====================

def end_game(pin: str):
    with games_lock:
        game = active_games.get(pin)
        if not game or game.status == 'finished':
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
        del active_games[pin]

    with app.app_context():
        history = GameHistory.query.filter_by(pin=pin).order_by(GameHistory.created_at.desc()).first()
        if history and history.ended_at is None:
            history.ended_at = datetime.utcnow()
            history.winner_team = winner
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
                        user.total_wins += 1
                    if mode == 'ffa' and p['name'] == winner:
                        user.total_wins += 1
                if user.total_games > 0:
                    user.rating = round((user.total_wins / user.total_games) * 100, 1)
            db.session.commit()

    socketio.emit('game_finished', {
        'leaderboard': leaderboard,
        'stats': stats,
        'topic': topic,
        'mode': mode,
        'winner': winner,
    }, room=pin)

def time_up(pin: str, timer_id: str):
    with games_lock:
        game = active_games.get(pin)
        if not game or game.status != 'playing' or game.question_timer_id != timer_id:
            return
        game.cancel_question_timer()
        if game.mode == 'ffa':
            for p in game.players.values():
                if not p.get('answered_current'):
                    p['wrong'] += 1
                    p['answered_current'] = True
    socketio.emit('time_up', room=pin)
    _schedule_next_question(pin)

def _schedule_next_question(pin: str):
    def _advance():
        with games_lock:
            game = active_games.get(pin)
            if not game or game.status != 'playing':
                return
            has_next = game.next_question()
        if not has_next:
            end_game(pin)
            return
        socketio.emit('next_question_ready', room=pin)
    t = threading.Timer(NEXT_QUESTION_DELAY, _advance)
    t.daemon = True
    t.start()

@socketio.on('connect')
def handle_connect():
    pass

@socketio.on('disconnect')
def handle_disconnect():
    with games_lock:
        for pin, game in list(active_games.items()):
            token = game.get_player_token_by_sid(request.sid)
            if not token:
                continue
            game.mark_disconnected(request.sid)
            def _cleanup_disconnect(pin=pin, player_token=token):
                removed_name = None
                players_payload = []
                with games_lock:
                    g = active_games.get(pin)
                    if not g:
                        return
                    p2 = g.players.get(player_token)
                    if not p2 or p2.get('connected'):
                        return
                    removed_name = g.remove_player(player_token)
                    players_payload = get_players_list(g)
                    if len(g.players) == 0:
                        del active_games[pin]
                if removed_name:
                    socketio.emit('player_left', {'name': removed_name, 'players': players_payload}, room=pin)
            timer = threading.Timer(DISCONNECT_GRACE_SECONDS, _cleanup_disconnect)
            timer.daemon = True
            p2 = game.players.get(token)
            if p2:
                p2['disconnect_timer'] = timer
            timer.start()
            break

def get_players_list(game):
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
    topic = (data.get('topic') or '').strip()
    mode = (data.get('mode') or 'teams').strip()
    difficulty = (data.get('difficulty') or 'medium').strip()
    questions_count = int(data.get('questions_count', 5))
    bonus_enabled = bool(data.get('bonus_enabled', True))
    creator_token = (data.get('player_token') or '').strip() or uuid.uuid4().hex

    if not topic:
        emit('error_message', {'message': 'укажи тему для игры'})
        return
    topic = topic[:50].lower()
    if mode not in ('teams', 'ffa'):
        mode = 'teams'
    if difficulty not in ('easy', 'medium', 'hard'):
        difficulty = 'medium'
    if mode == 'teams' and questions_count not in (5, 6, 7):
        questions_count = 5

    game = GameSession(
        creator_id=current_user.id if current_user.is_authenticated else None,
        creator_token=creator_token,
        topic=topic,
        mode=mode,
        difficulty=difficulty,
        questions_count=questions_count,
        time_limit=QUESTION_TIME_LIMIT,
        bonus_enabled=bonus_enabled,
    )

    with games_lock:
        active_games[game.pin] = game

    join_room(game.pin)
    emit('game_created', {
        'pin': game.pin,
        'topic': game.topic,
        'mode': game.mode,
        'difficulty': game.difficulty,
        'questions_count': game.questions_count,
        'questions_per_team': game.questions_per_team,
        'time_limit': game.time_limit,
    })

def _normalize_player_name(raw_name: str, player_token: str):
    name = (raw_name or '').strip()
    return name[:20] if name else f"игрок-{player_token[:4]}"

def _is_creator(game: GameSession, *, actor_token: str | None):
    if current_user.is_authenticated and game.creator_id and current_user.id == game.creator_id:
        return True
    if actor_token and game.creator_token and actor_token == game.creator_token:
        return True
    return False

@socketio.on('join_game')
def handle_join_game(data):
    pin = (data.get('pin') or '').upper().strip()
    player_token = (data.get('player_token') or '').strip()
    guest_name = data.get('guest_name', '')

    if not pin:
        emit('error_message', {'message': 'неверный pin'})
        return
    if not player_token:
        player_token = uuid.uuid4().hex

    with games_lock:
        game = active_games.get(pin)
        if not game:
            emit('error_message', {'message': 'игра не найдена'})
            return
        if game.status in ('playing', 'paused'):
            if player_token not in game.players:
                emit('error_message', {'message': 'игра уже началась'})
                return
        if game.status == 'finished':
            emit('error_message', {'message': 'игра уже закончилась'})
            return

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
            'bonus_enabled': game.bonus_enabled,   # ← добавлено
        }

    emit('joined', joined_payload, to=request.sid)
    socketio.emit('player_joined', {'name': final_name, 'players': joined_payload['players']}, room=pin)

@socketio.on('start_game')
def handle_start_game(data):
    pin = (data.get('pin') or '').upper().strip()
    actor_token = (data.get('player_token') or '').strip()

    with games_lock:
        game = active_games.get(pin)
        if not game or game.status != 'waiting':
            return
        if not _is_creator(game, actor_token=actor_token):
            emit('error_message', {'message': 'только ведущий может начать игру'})
            return
        if len(game.players) < 2:
            emit('error_message', {'message': 'нужно минимум 2 игрока'})
            return

        game.load_questions()
        if not game.questions or len(game.questions) < game.questions_count:
            emit('error_message', {'message': 'не удалось загрузить вопросы'})
            return

        game.status = 'playing'
        game.current_question_idx = 0
        game.current_team = 'A'
        game.round_answered = False
        game.cancel_question_timer()

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
    pin = (data.get('pin') or '').upper().strip()
    with games_lock:
        game = active_games.get(pin)
        if not game or game.status != 'playing':
            return
        player = game.get_player_by_sid(request.sid)
        if not player:
            emit('error_message', {'message': 'ты не в этой игре'})
            return
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
    pin = (data.get('pin') or '').upper().strip()
    answer = data.get('answer')
    try:
        answer = int(answer)
    except Exception:
        return

    with games_lock:
        game = active_games.get(pin)
        if not game or game.status != 'playing' or not game.question_start_time:
            return
        player = game.get_player_by_sid(request.sid)
        if not player:
            return

        if game.mode == 'teams':
            if player.get('team') != game.current_team or game.round_answered:
                return
        else:
            if player.get('answered_current'):
                return

        is_correct = game.check_answer(answer)
        response_time = time.time() - game.question_start_time
        points = game.calculate_score(is_correct=is_correct, response_time=response_time)

        player['answered_current'] = True
        player['response_times'].append(response_time)
        if is_correct:
            player['score'] += points
            player['correct'] += 1
        else:
            player['wrong'] += 1

        answered_by = player['name']
        if game.mode == 'teams':
            game.round_answered = True
            game.cancel_question_timer()
        leaderboard = game.get_leaderboard()
        current_team = game.current_team

        all_answered = False
        if game.mode == 'ffa':
            all_answered = all(p.get('answered_current') for p in game.players.values())
            if all_answered:
                game.cancel_question_timer()

    emit('answer_result', {'correct': is_correct, 'answer': answer, 'score': points}, to=request.sid)
    socketio.emit('score_update', {
        'leaderboard': leaderboard,
        'answered_by': answered_by,
        'is_correct': is_correct,
    }, room=pin)

    if game.mode == 'teams':
        socketio.emit('round_locked', {
            'team': current_team,
            'answered_by': answered_by,
            'correct': is_correct,
        }, room=pin)
        _schedule_next_question(pin)
        return

    if all_answered:
        _schedule_next_question(pin)

@socketio.on('admin_skip')
def handle_skip(data):
    pin = (data.get('pin') or '').upper().strip()
    actor_token = (data.get('player_token') or '').strip()
    with games_lock:
        game = active_games.get(pin)
        if not game or game.status != 'playing':
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
    pin = (data.get('pin') or '').upper().strip()
    actor_token = (data.get('player_token') or '').strip()
    with games_lock:
        game = active_games.get(pin)
        if not game:
            return
        if not _is_creator(game, actor_token=actor_token):
            return
    end_game(pin)

# ==================== API РОУТЫ ====================

@app.route('/api/game/<pin>/stats')
def get_game_stats(pin):
    with games_lock:
        if pin in active_games:
            game = active_games[pin]
            return jsonify(game.get_stats())
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
    """Экспорт результатов игры в текстовый файл"""
    filename = f'game_{pin}_{int(time.time())}.txt'
    # Кроссплатформенный путь к временной папке
    filepath = os.path.join(tempfile.gettempdir(), filename)
    data_lines = []


    with games_lock:
        if pin in active_games:
            game = active_games[pin]
            stats = game.get_stats()
            data_lines.append(f"Игра {pin}")
            data_lines.append(f"Тема: {game.topic}")
            data_lines.append(f"Режим: {'Команды' if game.mode == 'teams' else 'FFA'}")
            data_lines.append(f"Сложность: {game.difficulty}")
            data_lines.append(f"Дата: {datetime.fromtimestamp(game.created_at).strftime('%Y-%m-%d %H:%M')}")
            data_lines.append("")
            data_lines.append("РЕЗУЛЬТАТЫ:")
            for i, p in enumerate(stats, 1):
                team_str = f" [Команда {p['team']}]" if p.get('team') else ""
                data_lines.append(f"{i}. {p['name']}{team_str} – {p['score']} очков (правильно: {p['correct']}, ошибок: {p['wrong']})")
        else:
            history = GameHistory.query.filter_by(pin=pin).order_by(GameHistory.created_at.desc()).first()
            if not history:
                return jsonify({'error': 'игра не найдена'}), 404
            stats = PlayerStats.query.filter_by(game_id=history.id).all()
            data_lines.append(f"Игра {pin}")
            data_lines.append(f"Тема: {history.topic}")
            data_lines.append(f"Режим: {'Команды' if history.mode == 'teams' else 'FFA'}")
            data_lines.append(f"Сложность: {history.difficulty}")
            data_lines.append(f"Дата: {history.created_at.strftime('%Y-%m-%d %H:%M')}")
            if history.ended_at:
                data_lines.append(f"Окончена: {history.ended_at.strftime('%Y-%m-%d %H:%M')}")
            data_lines.append("")
            data_lines.append("РЕЗУЛЬТАТЫ:")
            for i, s in enumerate(stats, 1):
                name = s.guest_name or (s.user.username if s.user else 'Игрок')
                team_str = f" [Команда {s.team}]" if s.team else ""
                data_lines.append(f"{i}. {name}{team_str} – {s.score} очков (правильно: {s.correct_answers}, ошибок: {s.wrong_answers})")

    if not data_lines:
        return jsonify({'error': 'игра не найдена'}), 404

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(data_lines))

    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename,
        mimetype='text/plain'
    )

# ==================== ИНИЦИАЛИЗАЦИЯ ====================

def init_db():
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
#!/usr/bin/env python3
"""
десктопная версия quizup
запускает сервер и открывает окно приложения
"""

import threading
import time
import webview
from app import app, socketio, init_db

# флаг для отслеживания запуска сервера
server_ready = False


def start_server():
    """запуск flask сервера в отдельном потоке"""
    # инициализируем базу данных
    init_db()
    
    # запускаем сервер
    print("[*] запуск сервера на http://127.0.0.1:5000")
    server_ready = True
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False)


def main():
    """основная функция десктопного приложения"""
    # запускаем сервер в фоновом потоке
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # ждем пока сервер запустится
    print("[*] ожидание запуска сервера...")
    time.sleep(2)
    
    # создаем окно приложения
    print("[*] открытие окна приложения")
    window = webview.create_window(
        title='QuizUp - Командная Викторина',
        url='http://127.0.0.1:5000',
        width=1200,
        height=800,
        min_size=(800, 600),
        resizable=True,
        text_select=True
    )
    
    # запускаем GUI
    webview.start(debug=False)


if __name__ == '__main__':
    main()

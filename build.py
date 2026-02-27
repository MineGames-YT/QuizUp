#!/usr/bin/env python3
"""
скрипт сборки exe файла
запуск: python build.py
"""

import PyInstaller.__main__
import os
import shutil

# очищаем старые сборки
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('build'):
    shutil.rmtree('build')

print("[*] начинаем сборку...")
print("[*] это может занять несколько минут...")

PyInstaller.__main__.run([
    'desktop.py',
    '--name=QuizBattle',
    '--onefile',
    '--windowed',
    '--add-data=templates:templates',
    '--add-data=static:static',
    '--icon=NONE',
    '--clean',
    '--noconfirm'
])

print("[*] сборка завершена!")
print("[*] файл находится в папке dist/")

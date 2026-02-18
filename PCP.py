"""
People Counter Pro - Профессиональная система видеоаналитики
==========================================================

Многофункциональное приложение для подсчета людей и анализа трафика.
Основные возможности:
- Детекция и трекинг через YOLOv8 (Medium/Nano)
- Аппаратное ускорение (CUDA/FP16)
- Динамическое размытие лиц (Анонимизация)
- Тепловые карты интенсивности (Heatmaps)
- Интеграция с Google Drive и Email-уведомления
- Система приоритетов процесса для максимальной производительности

Разработчик: Antigravity AI
Версия: 3.0 Pro
"""
import cv2  # OpenCV: Библиотека компьютерного зрения для обработки изображений и видео
import sys  # System: Для взаимодействия с интерпретатором Python и системой
import ctypes  # Ctypes: Для вызова функций из динамических библиотек (например, настройка DPI в Windows)
import numpy as np  # NumPy: Для работы с массивами и матрицами (обработка кадров)
from ultralytics import YOLO  # Ultralytics YOLO: Нейросеть для обнаружения и отслеживания объектов
import torch  # Torch: Для проверки доступности GPU и управления устройством
import tkinter as tk  # Tkinter: Стандартная библиотека для создания GUI
from tkinter import ttk, messagebox, filedialog, simpledialog  # Виджеты и диалоги Tkinter
import json  # JSON: Для работы с файлами конфигурации и данных
import csv  # CSV: Для сохранения отчетов в табличном формате
import time  # Time: Для работы со временем (задержки, таймеры)
import os  # OS: Для работы с файловой системой (пути, файлы)
from datetime import datetime  # Datetime: Для работы с датами и временем
import threading  # Threading: Для многопоточности (чтобы интерфейс не зависал при обработке видео)
import math  # Math: Математические функции (расчет расстояний, геометрия)
from PIL import Image, ImageTk, ImageDraw, ImageFont  # Pillow: Библиотека для работы с изображениями (отображение в GUI, рисование текста)
import smtplib  # Smtplib: Для отправки email уведомлений
from email.mime.text import MIMEText  # Email: Формирование текста письма
from email.mime.multipart import MIMEMultipart  # Email: Формирование составных писем (с вложениями)
import requests  # Requests: Для HTTP запросов (если потребуется)
import ssl  # SSL: Для безопасного соединения (HTTPS/SMTP)
import configparser  # Configparser: Для работы с ini файлами (в данном коде используется json, но импорт оставлен)
import hashlib  # Hashlib: Для хеширования данных
from cryptography.fernet import Fernet  # Cryptography: Для шифрования конфиденциальных данных (паролей, токенов)
import psutil  # Psutil: Для управления процессами (установка приоритета)

# Google Drive API импорты
from google.oauth2.credentials import Credentials  # Управление учетными данными OAuth
from google_auth_oauthlib.flow import InstalledAppFlow  # Процесс авторизации пользователя
from google.auth.transport.requests import Request  # Обновление токенов
from googleapiclient.discovery import build  # Создание клиента API
from googleapiclient.http import MediaFileUpload  # Загрузка файлов
import webbrowser  # Webbrowser: Для открытия ссылок в браузере
import pickle  # Pickle: Для сериализации объектов (сохранение токенов)

# Область доступа к Google Drive (только создание и редактирование файлов, созданных приложением)
SCOPES = ['https://www.googleapis.com/auth/drive.file']
# Базовая директория приложения (где находится скрипт)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# ЯЗЫКОВАЯ ЛОКАЛИЗАЦИЯ
# ==========================================
LANGUAGES = {
    "en": {
        "app_title": "People Counter Pro v3.0 (Analytics & Privacy)",
        "video_source": "🎥 Video Source",
        "webcam": "Webcam (USB)",
        "ip_camera": "IP Camera (RTSP/HTTP)",
        "video_file": "Video File",
        "select_file": "Select File",
        "timer": "⏳ Runtime Timer",
        "hours": "Hours:",
        "minutes": "Min:",
        "system_settings": "⚙️ System Settings",
        "record_video": "Record video (MP4)",
        "save_csv": "Save CSV report",
        "debug_mode": "Debug mode (ID/Centers)",
        "anti_duplication": "Anti-duplication (sec):",
        "heatmap": "🔥 Heatmap",
        "save_heatmap_img": "Save heatmap image",
        "heatmap_params": "🔧 Heatmap Parameters",
        "google_drive": "☁️ Google Drive",
        "gdrive_config": "⚙️ Configure Google Drive",
        "notifications": "🔔 Notification System",
        "notif_config": "⚙️ Configure Notifications",
        "save_zones": "💾 Save Zones",
        "load_zones": "📂 Load Zones",
        "start_analysis": "▶ START ANALYSIS",
        "help_text": (
            "Controls during capture:\n"
            "• LMB: Draw line/zone  |  RMB: Delete last line/zone\n"
            "• [1]: ENTRY mode (Green)\n"
            "• [2]: EXIT mode (Red)\n"
            "• [3]: CONTROL mode (Yellow)\n"
            "• [4]: INTEREST ZONE mode (Purple) – place points sequentially,\n"
            "      click on the first point to close the area\n"
            "• [5]: Toggle HEATMAP (if enabled)\n"
            "• [S]: Save  |  [L]: Load  |  [ESC]: Exit"
        ),
        "gdrive_not_configured": "❌ Not configured",
        "gdrive_active": "✅ Active",
        "gdrive_auth_needed": "⚠️ Auth required",
        "email_off": "📧❌",
        "email_on": "📧✅",

        "language": "Language",
        "restart_to_apply": "Please restart the application to apply language changes.",
        "error": "Error",
        "success": "Success",
        "zones_saved": "Zones successfully saved to file:\n{}",
        "file_load_error": "Failed to load file:\n{}",
        "json_corrupted": "File is corrupted or invalid JSON format",
        "no_video_selected": "No video file selected!",
        "no_contacts_warning": (
            "Notifications are enabled but no contacts specified!\n"
            "Continue without notifications?"
        ),
        "processing_aborted": "Video processing canceled by user",
        "processing_complete": "✅ Processing completed successfully!",
        "video_timelapse": "📹 Timelapse: {}",
        "csv_report": "📊 Report: {}",
        "csv_disabled": "📊 CSV report NOT saved (disabled in settings)",
        "heatmap_saved": "🔥 Heatmap: {}",
        "gdrive_uploaded": "\n☁️ Files uploaded to Google Drive!",
        "notifications_sent": "\n🔔 Notifications sent: {}",
        "processing_time": "Processing time: {}",
        "frames_processed": "Frames processed: {}",
        "lines_added": "Lines/zones added: {}",
        "stats": (
            "Statistics:\n"
            "  Entry: {}\n"
            "  Exit: {}\n"
            "  Control: {}\n"
            "  Zones: {}\n"
            "  Total on frame: {}"
        ),
        "confirm_markup": "Markup confirmed!\nAdded elements: {}",
        "markup_window_title": "Markup lines and zones on first frame",
        "confirm_button": "Confirm (Enter)",
        "cancel_button": "Cancel (ESC)",
        "mode_label": "Mode: {} | Elements: {}",
        "heatmap_label_on": "Heatmap: ON",
        "heatmap_label_off": "Heatmap: OFF",
        "entry": "ENTRY",
        "exit": "EXIT",
        "control": "CONTROL",
        "interest_zone": "INTEREST ZONE",
        "zone_id": "ID",
        "zone_table_header": "ZONE COUNTERS",
        "count": "Count",
        "color": "Color",
        "in_count": "IN: {}",
        "out_count": "OUT: {}",
        "ctrl_count": "CTRL: {}",
        "zone_count": "ZONE: {}",
        "total_count": "TOTAL: {}",
        "current_time": "TIME: {}",
        "orange": "Orange",
        "mode_display": "MODE: {}",
        "heatmap_transparency": "Transparency (0.0 - 1.0):",
        "heatmap_blur": "Blur size (odd):",
        "heatmap_decay": "Decay (0.90 - 1.00):",
        "save": "Save",
        "cancel": "Cancel",
        "settings_saved": "Settings saved",
        "heatmap_settings_applied": "Heatmap settings applied!",
        "invalid_value": "Invalid value: {}",
        "email_tab": "📧 Email",

        "thresholds_contacts_tab": "📊 Thresholds & Contacts",
        "smtp_server": "SMTP Server:",
        "port": "Port:",
        "email_label": "Email:",
        "password_label": "Password:",
        "show_password": "Show password",
        "enable_email": "Enable email notifications",

        "threshold_values": "Threshold values",
        "threshold_label": "{} threshold:",
        "contacts_label": "Notification Contacts",
        "email_addresses": "Email addresses (comma separated):",
        "phones_label": "Phones (comma separated, +7...):",
        "test_email": "🧪 Test Email",

        "test_email_success": "Email test successful!\n{}",
        "connection_error": "Connection error:\n{}",

        "config_saved": "Notification configuration saved successfully!",
        "processing_video": "Processing video...",
        "optimization_title": "Processing Optimization",
        "optimization_msg": "Recommended to reduce quality to 640x480 and 30 FPS for faster processing.\nThis will reduce time by 3-5 times! Otherwise, it may take hours.\nApply optimization?",
        "video_open_error": "Failed to open video file",
        "first_frame_error": "Failed to read first frame",
        "time_est_title": "Time Estimation",
        "total_frames": "Total frames: {}",
        "frames_to_process": "Frames to process: {}",
        "est_time": "Estimated time: {}",
        "continue_q": "Continue processing?",
        "always_on_top": "Always on top",
        "files_saved_title": "Files saved successfully:",
        "video_source_error": "Failed to open video source",
        "connection_status": "Connection Status",
        "not_connected": "❌ Not connected",
        "folder": "Folder: {}",
        "auth_btn": "🔐 Authenticate",
        "create_folder_btn": "📁 Create Folder",
        "open_folder_btn": "🔗 Open Folder",
        "upload_settings": "Upload Settings",
        "auto_upload": "Automatically upload files to Google Drive",
        "folder_id_label": "Upload Folder (Link or ID):",
        "instructions_title": "Instructions",
        "instructions_text": (
            "1. Click 'Authenticate' to connect to Google Drive\n"
            "2. Create or select an existing folder for file uploads\n"
            "3. Enable automatic file upload\n"
            "4. Files will be automatically uploaded after processing\n"
            "Supported files:\n"
            "- Video files (*.mp4)\n"
            "- CSV reports (*.csv)\n"
            "- Heatmaps (*.png)"
        ),
        "connected": "✅ Connected to Google Drive",
        "folder_not_selected": "Folder: Not selected",
        "auth_success": "Successfully connected to Google Drive!",
        "auth_fail": "Failed to connect to Google Drive",
        "auth_common_error": "Authenticate with Google Drive first",
        "folder_name_prompt": "Enter folder name for file uploads:",
        "folder_create_success": "Folder '{}' created successfully!\nFolder ID: {}",
        "folder_create_fail": "Failed to create folder",
        "folder_select_error": "Folder not selected. Create or specify folder ID first.",
        "config_saved_gdrive": "Google Drive configuration saved successfully!",
        "folder_create_title": "Create Folder",
        "info_title": "Information",

        "thresholds_tab": "Thresholds",
        "notif_msg_threshold": "Threshold exceeded for {}:\nCurrent value: {}\nThreshold: {}",
        "notif_subject_threshold": "🚨 THRESHOLD EXCEEDED: {} = {}",

        "total": "Total",

        "threshold_label_suffix": " Threshold:",
        "contacts_frame_title": "Notification Contacts",
        "emails_label": "Email addresses (comma separated):",
        "phones_label": "Phone numbers (comma separated, format +7...):",
        "copy": "Copy",
        "paste": "Paste",
        "cut": "Cut",
        "select_all": "Select All",
        "face_blur": "Blur Faces (Anon)",
        "gpu_acceleration": "GPU Acceleration (CUDA)",
        "turbo_mode": "Turbo Mode (MAX Load & Accuracy)",
        "email_notif_header": "🚨 THRESHOLD EXCEEDED NOTIFICATION",
        "email_notif_time": "Time:",
        "email_notif_stats": "Current statistics:",
        "email_notif_footer": "This notification was sent by an automated attendance monitoring system.",
        "threshold_simple": "threshold",
        "connection_success": "Connection successful!",
        "smtp_server_label": "SMTP Server:",
        "port_label": "Port:",
        "email_label": "Email:",
        "entered_label": "ENTRY",
        "exited_label": "EXIT",
        "control_label": "CONTROL",
        "zone_label": "INTEREST ZONE",
        "total_label": "TOTAL",
        "hours_unit": "h",
        "mins_unit": "min",
        "secs_unit": "sec",
        "folder_link": "🔗 Folder link: {}",
        "files_saved_label": "Files saved successfully:"
    },
    "ru": {
        "app_title": "People Counter Pro v3.0 (Аналитика и Приватность)",
        "video_source": " 🎥 Источник видео ",
        "webcam": "Веб-камера (USB)",
        "ip_camera": "IP Камера (RTSP/HTTP)",
        "video_file": "Видеофайл",
        "select_file": "Выбрать файл",
        "timer": " ⏳ Таймер работы ",
        "hours": "Часы:",
        "minutes": "Мин:",
        "system_settings": " ⚙️ Настройки системы ",
        "record_video": "Записывать видео (MP4)",
        "save_csv": "Сохранять CSV отчет",
        "debug_mode": "Режим отладки (ID/Центры)",
        "anti_duplication": "Анти-дублирование (сек):",
        "heatmap": " 🔥 Тепловая карта ",
        "save_heatmap_img": "Сохранить изображение тепловой карты",
        "heatmap_params": "🔧 Параметры тепловой карты",
        "google_drive": " ☁️ Google Drive ",
        "gdrive_config": "⚙️ Настроить Google Drive",
        "notifications": " 🔔 Система оповещений ",
        "notif_config": "⚙️ Настроить уведомления",
        "save_zones": "💾 Сохранить зоны",
        "load_zones": "📂 Загрузить зоны",
        "start_analysis": "▶ ЗАПУСТИТЬ АНАЛИЗ",
        "help_text": (
            "Управление во время съемки:\n"
            "• ЛКМ: Рисовать линию/зону  |  ПКМ: Удалить последнюю линию/зону\n"
            "• Клавиша [1]: Режим ВХОД (Зеленый)\n"
            "• Клавиша [2]: Режим ВЫХОД (Красный)\n"
            "• Клавиша [3]: Режим КОНТРОЛЬ (Желтый)\n"
            "• Клавиша [4]: Режим ЗОНА ИНТЕРЕСА (Фиолетовый) - последовательно ставьте точки,\n"
            "             кликните на первую точку для замыкания области\n"
            "• Клавиша [5]: Вкл/Выкл ТЕПЛОВАЯ КАРТА (если включена)\n"
            "• Клавиша [S]: Сохранить  |  [L]: Загрузить  |  [ESC]: Выход"
        ),
        "gdrive_not_configured": "❌ Не настроен",
        "gdrive_active": "✅ Активен",
        "gdrive_auth_needed": "⚠️ Требуется авторизация",
        "email_off": "📧❌",
        "email_on": "📧✅",

        "language": "Язык",
        "restart_to_apply": "Пожалуйста, перезапустите приложение для применения языка.",
        "error": "Ошибка",
        "success": "Успех",
        "zones_saved": "Зоны успешно сохранены в файл:\n{}",
        "file_load_error": "Не удалось загрузить файл:\n{}",
        "json_corrupted": "Файл поврежден или имеет неверный формат JSON",
        "no_video_selected": "Не выбран видеофайл!",
        "no_contacts_warning": (
            "Включены уведомления, но не указаны контакты!\n"
            "Продолжить без уведомлений?"
        ),
        "processing_aborted": "Обработка видео отменена пользователем",
        "processing_complete": "✅ Обработка завершена успешно!",
        "video_timelapse": "📹 Таймлапс: {}",
        "csv_report": "📊 Отчет: {}",
        "csv_disabled": "📊 CSV отчет НЕ сохранялся (отключено в настройках)",
        "heatmap_saved": "🔥 Тепловая карта: {}",
        "gdrive_uploaded": "\n☁️ Файлы успешно загружены в Google Drive!",
        "notifications_sent": "\n🔔 Отправлено уведомлений: {}",
        "processing_time": "Время обработки: {}",
        "frames_processed": "Обработано кадров: {}",
        "lines_added": "Добавлено линий/зон: {}",
        "stats": (
            "Статистика:\n"
            "  Вход: {}\n"
            "  Выход: {}\n"
            "  Контроль: {}\n"
            "  Зоны: {}\n"
            "  Всего на кадре: {}"
        ),
        "confirm_markup": "Разметка подтверждена!\nДобавлено элементов: {}",
        "markup_window_title": "Разметка линий и зон на первом кадре",
        "confirm_button": "Подтвердить (Enter)",
        "cancel_button": "Отменить (ESC)",
        "mode_label": "Режим: {} | Элементов: {}",
        "heatmap_label_on": "Тепловая карта: ВКЛ",
        "heatmap_label_off": "Тепловая карта: ВЫКЛ",
        "entry": "ВХОД",
        "exit": "ВЫХОД",
        "control": "КОНТРОЛЬ",
        "interest_zone": "ЗОНА ИНТЕРЕСА",
        "zone_id": "ID",
        "zone_table_header": "СЧЕТЧИКИ ЗОН",
        "count": "Счет",
        "color": "Цвет",
        "in_count": "ВХОД: {}",
        "out_count": "ВЫХОД: {}",
        "ctrl_count": "КОНТРОЛЬ: {}",
        "zone_count": "ЗОНА: {}",
        "total_count": "ВСЕГО: {}",
        "current_time": "ВРЕМЯ: {}",
        "orange": "Оранжевый",
        "mode_display": "РЕЖИМ: {}",
        "heatmap_transparency": "Прозрачность (0.0 - 1.0):",
        "heatmap_blur": "Размер размытия (нечетное):",
        "heatmap_decay": "Затухание (0.90 - 1.00):",
        "save": "Сохранить",
        "cancel": "Отмена",
        "settings_saved": "Параметры сохранены",
        "heatmap_settings_applied": "Настройки тепловой карты применены!",
        "invalid_value": "Некорректное значение: {}",
        "email_tab": "📧 Email",

        "thresholds_contacts_tab": "📊 Пороги и контакты",
        "smtp_server": "SMTP Сервер:",
        "port": "Порт:",
        "email_label": "Email:",
        "password_label": "Пароль:",
        "show_password": "Показать пароль",
        "enable_email": "Включить email уведомления",

        "threshold_values": "Пороговые значения",
        "threshold_label": "{} порог:",
        "contacts_label": "Контакты для уведомлений",
        "email_addresses": "Email адреса (через запятую):",
        "phones_label": "Телефоны (через запятую, формат +7...):",
        "test_email": "🧪 Тест Email",

        "test_email_success": "Тест email успешен!\n{}",
        "connection_error": "Ошибка подключения:\n{}",

        "config_saved": "Конфигурация оповещений успешно сохранена!",
        "processing_video": "Обработка видео...",
        "optimization_title": "Оптимизация обработки",
        "optimization_msg": "Рекомендуется снизить качество видео до 640x480 и 30 FPS для ускорения обработки.\nЭто сократит время обработки в 3-5 раз! Иначе обработка может занять часы.\nПрименить оптимизацию?",
        "video_open_error": "Не удалось открыть видеофайл",
        "first_frame_error": "Не удалось прочитать первый кадр",
        "time_est_title": "Оценка времени обработки",
        "total_frames": "Общее количество кадров: {}",
        "frames_to_process": "Кадров для обработки: {}",
        "est_time": "Предполагаемое время обработки: {}",
        "continue_q": "Продолжить обработку?",
        "always_on_top": "Поверх других окон",
        "files_saved_title": "Файлы успешно сохранены:",
        "video_source_error": "Не удалось открыть видеоисточник",
        "connection_status": "Статус подключения",
        "not_connected": "❌ Не подключено",
        "folder": "Папка: {}",
        "auth_btn": "🔐 Авторизоваться",
        "create_folder_btn": "📁 Создать папку",
        "open_folder_btn": "🔗 Открыть папку",
        "upload_settings": "Настройки загрузки",
        "auto_upload": "Автоматически загружать файлы в Google Drive",
        "folder_id_label": "Папка для загрузки (Ссылка или ID):",
        "instructions_title": "Инструкция",
        "instructions_text": (
            "1. Нажмите 'Авторизоваться' для подключения к Google Drive\n"
            "2. Создайте или выберите существующую папку для загрузки файлов\n"
            "3. Включите автоматическую загрузку файлов\n"
            "4. После обработки видео файлы будут автоматически загружены\n"
            "Поддерживаемые файлы:\n"
            "- Видео файлы (*.mp4)\n"
            "- CSV отчеты (*.csv)\n"
            "- Тепловые карты (*.png)"
        ),
        "connected": "✅ Подключено к Google Drive",
        "folder_not_selected": "Папка: Не выбрана",
        "auth_success": "Успешное подключение к Google Drive!",
        "auth_fail": "Не удалось подключиться к Google Drive",
        "auth_common_error": "Сначала авторизуйтесь в Google Drive",
        "folder_name_prompt": "Введите название папки для загрузки файлов:",
        "folder_create_success": "Папка '{}' создана успешно!\nID папки: {}",
        "folder_create_fail": "Не удалось создать папку",
        "folder_select_error": "Папка не выбрана. Сначала создайте или укажите ID папки.",
        "config_saved_gdrive": "Конфигурация Google Drive успешно сохранена!",
        "folder_create_title": "Создать папку",
        "info_title": "Информация",

        "thresholds_tab": "Пороговые значения",
        "notif_msg_threshold": "Обнаружено превышение порогового значения для {}:\nТекущее значение: {}\nПорог: {}",
        "notif_subject_threshold": "🚨 ПРЕВЫШЕН ПОРОГ: {} = {}",

        "total": "Всего",

        "threshold_label_suffix": " порог:",
        "contacts_frame_title": "Контакты для уведомлений",
        "emails_label": "Email адреса (через запятую):",
        "phones_label": "Телефоны (через запятую, формат +7...):",
        "copy": "Копировать",
        "paste": "Вставить",
        "cut": "Вырезать",
        "select_all": "Выделить всё",
        "face_blur": "Размытие лиц (Аноним)",
        "gpu_acceleration": "GPU ускорение (CUDA)",
        "turbo_mode": "Турбо-режим (MAX точность и нагрузка)",
        "email_notif_header": "🚨 УВЕДОМЛЕНИЕ О ПРЕВЫШЕНИИ ПОРОГА",
        "email_notif_time": "Время:",
        "email_notif_stats": "Текущая статистика:",
        "email_notif_footer": "Это уведомление отправлено автоматической системой мониторинга посещаемости.",
        "threshold_simple": "порог",
        "connection_success": "Подключение успешно!",
        "smtp_server_label": "SMTP Сервер:",
        "port_label": "Порт:",
        "email_label": "Email:",
        "entered_label": "ВХОД",
        "exited_label": "ВЫХОД",
        "control_label": "КОНТРОЛЬ",
        "zone_label": "ЗОНА ИНТЕРЕСА",
        "total_label": "ВСЕГО",
        "hours_unit": "ч",
        "mins_unit": "мин",
        "secs_unit": "сек",
        "folder_link": "🔗 Ссылка на папку: {}",
        "files_saved_label": "Файлы успешно сохранены:"
    },
    "kk": {
        "app_title": "People Counter Pro v3.0 (Талдау және Құпиялылық)",
        "video_source": " 🎥 Бейне көзі ",
        "webcam": "Веб-камера (USB)",
        "ip_camera": "IP камера (RTSP/HTTP)",
        "video_file": "Бейне файлы",
        "select_file": "Файлды таңдау",
        "timer": " ⏳ Жұмыс уақыты ",
        "hours": "Сағат:",
        "minutes": "Мин:",
        "system_settings": " ⚙️ Жүйе параметрлері ",
        "record_video": "Бейнені жазу (MP4)",
        "save_csv": "CSV есеп беруді сақтау",
        "debug_mode": "Жөндеу режимі (ID/Орталар)",
        "anti_duplication": "Қайталанудан қорғау (сек):",
        "heatmap": " 🔥 Жылу картасы ",
        "save_heatmap_img": "Жылу картасының суретін сақтау",
        "heatmap_params": "🔧 Жылу картасы параметрлері",
        "google_drive": " ☁️ Google Drive ",
        "gdrive_config": "⚙️ Google Drive баптау",
        "notifications": " 🔔 Хабарландыру жүйесі ",
        "notif_config": "⚙️ Хабарландыруларды баптау",
        "save_zones": "💾 Аймақтарды сақтау",
        "load_zones": "📂 Аймақтарды жүктеу",
        "start_analysis": "▶ ТАЛДАУДЫ БАСТАУ",
        "help_text": (
            "Түсіру кезіндегі басқару:\n"
            "• Сол жақ тышқан: сызық/аймақ салу  |  Оң жақ тышқан: соңғы сызық/аймақты жою\n"
            "• [1] пернесі: КІРУ режимі (Жасыл)\n"
            "• [2] пернесі: ШЫҒУ режимі (Қызыл)\n"
            "• [3] пернесі: БАҚЫЛАУ режимі (Сары)\n"
            "• [4] пернесі: ҚЫЗЫҒУШЫЛЫҚ АЙМАҒЫ режимі (Күлгін) – нүктелерді ретімен қойыңыз,\n"
            "             аймақты жабу үшін бірінші нүктеге шертіңіз\n"
            "• [5] пернесі: ЖЫЛУ КАРТАСЫН қосу/сөндіру (егер қосылған болса)\n"
            "• [S]: Сақтау  |  [L]: Жүктеу  |  [ESC]: Шығу"
        ),
        "gdrive_not_configured": "❌ Бапталмаған",
        "gdrive_active": "✅ Белсенді",
        "gdrive_auth_needed": "⚠️ Авторизация қажет",
        "email_off": "📧❌",
        "email_on": "📧✅",

        "language": "Тіл",
        "restart_to_apply": "Тілді қолдану үшін қолданбаны қайта іске қосыңыз.",
        "error": "Қате",
        "success": "Сәтті",
        "zones_saved": "Аймақтар файлға сәтті сақталды:\n{}",
        "file_load_error": "Файлды жүктеу сәтсіз аяқталды:\n{}",
        "json_corrupted": "Файл зақымдалған немесе JSON пішімі дұрыс емес",
        "no_video_selected": "Бейне файлы таңдалмады!",
        "no_contacts_warning": (
            "Хабарландырулар қосылған, бірақ байланыс ақпараты көрсетілмеген!\n"
            "Хабарландыруларсыз жалғастырасыз ба?"
        ),
        "processing_aborted": "Бейнені өңдеу пайдаланушымен тоқтатылды",
        "processing_complete": "✅ Өңдеу сәтті аяқталды!",
        "video_timelapse": "📹 Уақыт өткен сайынғы бейне: {}",
        "csv_report": "📊 Есеп беру: {}",
        "csv_disabled": "📊 CSV есеп беру САҚТАЛМАДЫ (параметрлерде сөндірілген)",
        "heatmap_saved": "🔥 Жылу картасы: {}",
        "gdrive_uploaded": "\n☁️ Файлдар Google Drive-қа жүктелді!",
        "notifications_sent": "\n🔔 Хабарландырулар жіберілді: {}",
        "processing_time": "Өңдеу уақыты: {}",
        "frames_processed": "Өңделген кадрлар: {}",
        "lines_added": "Қосылған сызықтар/аймақтар: {}",
        "stats": (
            "Статистика:\n"
            "  Кіру: {}\n"
            "  Шығу: {}\n"
            "  Бақылау: {}\n"
            "  Аймақтар: {}\n"
            "  Кадрда барлығы: {}"
        ),
        "confirm_markup": "Белгілеу расталды!\nҚосылған элементтер: {}",
        "markup_window_title": "Бірінші кадрда сызықтар мен аймақтарды белгілеу",
        "confirm_button": "Растау (Enter)",
        "cancel_button": "Болдырмау (ESC)",
        "mode_label": "Режим: {} | Элементтер: {}",
        "heatmap_label_on": "Жылу картасы: ҚОСУЛҒАН",
        "heatmap_label_off": "Жылу картасы: СӨНДІРІЛГЕН",
        "entry": "КІРУ",
        "exit": "ШЫҒУ",
        "control": "БАҚЫЛАУ",
        "interest_zone": "ҚЫЗЫҒУШЫЛЫҚ АЙМАҒЫ",
        "zone_id": "ID",
        "zone_table_header": "АЙМАҚТАРДЫҢ САНАУЛАРЫ",
        "count": "Есеп",
        "color": "Түс",
        "in_count": "КІРУ: {}",
        "out_count": "ШЫҒУ: {}",
        "ctrl_count": "БАҚЫЛАУ: {}",
        "zone_count": "АЙМАҚ: {}",
        "total_count": "БАРЛЫҒЫ: {}",
        "current_time": "УАҚЫТ: {}",
        "orange": "Қызғылт сары",
        "mode_display": "РЕЖИМ: {}",
        "heatmap_transparency": "Мөлдірлік (0.0 - 1.0):",
        "heatmap_blur": "Бұлдырлық өлшемі (тақ сан):",
        "heatmap_decay": "Өшу (0.90 - 1.00):",
        "save": "Сақтау",
        "cancel": "Болдырмау",
        "settings_saved": "Параметрлер сақталды",
        "heatmap_settings_applied": "Жылу картасының параметрлері қолданылды!",
        "invalid_value": "Жарамсыз мән: {}",
        "email_tab": "📧 Email",

        "thresholds_contacts_tab": "📊 Табалдырықтар мен Контактілер",
        "smtp_server": "SMTP Сервер:",
        "port": "Порт:",
        "email_label": "Email:",
        "password_label": "Құпия сөз:",
        "show_password": "Құпия сөзді көрсету",
        "enable_email": "Email хабарландыруларды қосу",

        "threshold_values": "Табалдырық мәндері",
        "threshold_label": "{} табалдырығы:",
        "contacts_label": "Хабарландыру контактілері",
        "email_addresses": "Email мекенжайлары (үтір арқылы):",
        "phones_label": "Телефондар (үтір арқылы, +7...):",
        "test_email": "🧪 Email сынау",

        "test_email_success": "Email сынағы сәтті өтті!\n{}",
        "connection_error": "Қосылым қатесі:\n{}",

        "config_saved": "Хабарландыру конфигурациясы сәтті сақталды!",
        "processing_video": "Бейне өңделуде...",
        "optimization_title": "Өңдеуді оңтайландыру",
        "optimization_msg": "Өңдеуді тездету үшін бейне сапасын 640x480 және 30 FPS-ке дейін төмендету ұсынылады.\nБұл уақытты 3-5 есе қысқартады! Әйтпесе, өңдеу бірнеше сағатқа созылуы мүмкін.\nОңтайландыруды қолдану керек пе?",
        "video_open_error": "Бейне файлын ашу мүмкін болмады",
        "first_frame_error": "Бірінші кадрды оқу мүмкін болмады",
        "time_est_title": "Уақытты бағалау",
        "total_frames": "Барлық кадрлар: {}",
        "frames_to_process": "Өңделетін кадрлар: {}",
        "est_time": "Болжалды уақыт: {}",
        "continue_q": "Өңдеуді жалғастыру керек пе?",
        "always_on_top": "Басқа терезелердің үстінде",
        "files_saved_title": "Файлдар сәтті сақталды:",
        "video_source_error": "Бейне көзі ашылмады",
        "connection_status": "Қосылым күйі",
        "not_connected": "❌ Қосылмаған",
        "folder": "Папка: {}",
        "auth_btn": "🔐 Авторизациялау",
        "create_folder_btn": "📁 Папка құру",
        "open_folder_btn": "🔗 Папканы ашу",
        "upload_settings": "Жүктеу параметрлері",
        "auto_upload": "Файлдарды Google Drive-қа автоматты түрде жүктеу",
        "folder_id_label": "Жүктеу папкасы (Сілтеме немесе ID):",
        "instructions_title": "Нұсқаулық",
        "instructions_text": (
             "1. Google Drive-қа қосылу үшін 'Авторизациялау' түймесін басыңыз\n"
             "2. Файлдарды жүктеу үшін папка жасаңыз немесе таңдаңыз\n"
             "3. Файлдарды автоматты түрде жүктеуді қосыңыз\n"
             "4. Бейне өңделгеннен кейін файлдар автоматты түрде жүктеледі\n"
             "Қолдау көрсетілетін файлдар:\n"
             "- Бейне файлдар (*.mp4)\n"
             "- CSV есептері (*.csv)\n"
             "- Жылу карталары (*.png)"
        ),
        "connected": "✅ Google Drive-қа қосылды",
        "folder_not_selected": "Папка: Таңдалмаған",
        "auth_success": "Google Drive-қа сәтті қосылды!",
        "auth_fail": "Google Drive-қа қосылу мүмкін болмады",
        "auth_common_error": "Алдымен Google Drive-қа авторизацияланыңыз",
        "folder_name_prompt": "Файлдарды жүктеу үшін папка атауын енгізіңіз:",
        "folder_create_success": "'{}' папкасы сәтті құрылды!\nПапка ID: {}",
        "folder_create_fail": "Папка құру мүмкін болмады",
        "folder_select_error": "Папка таңдалмады. Алдымен папка ID-ін жасаңыз немесе көрсетіңіз.",
        "config_saved_gdrive": "Google Drive конфигурациясы сәтті сақталды!",
        "folder_create_title": "Папка құру",
        "info_title": "Ақпарат",

        "thresholds_tab": "Табалдырық мәндері",
        "notif_msg_threshold": "{} үшін табалдырық мәні асып кетті:\nАғымдағы мән: {}\nТабалдырық: {}",
        "notif_subject_threshold": "🚨 ТАБАЛДЫРЫҚ АСЫП КЕТТІ: {} = {}",

        "total": "Барлығы",

        "threshold_label_suffix": " табалдырығы:",
        "contacts_frame_title": "Хабарландыру контактілері",
        "emails_label": "Email мекенжайлары (үтір арқылы):",
        "phones_label": "Телефон нөмірлері (үтір арқылы, пішім +7...):",
        "copy": "Көшіру",
        "paste": "Қою",
        "cut": "Қиып алу",
        "select_all": "Барлығын таңдау",
        "face_blur": "Беттерді бұлдырату (Анон)",
        "gpu_acceleration": "GPU үдетуі (CUDA)",
        "email_notif_header": "🚨 ТАБАЛДЫРЫҚТАН АСУ ТУРАЛЫ ХАБАРЛАМА",
        "email_notif_time": "Уақыты:",
        "email_notif_stats": "Ағымдағы статистика:",
        "email_notif_footer": "Бұл хабарландыру автоматты келуді бақылау жүйесімен жіберілді.",
        "threshold_simple": "табалдырық",
        "connection_success": "Қосылу сәтті аяқталды!",
        "smtp_server_label": "SMTP сервері:",
        "port_label": "Порт:",
        "email_label": "Email:",
        "entered_label": "КІРУ",
        "exited_label": "ШЫҒУ",
        "control_label": "БАҚЫЛАУ",
        "zone_label": "ҚЫЗЫҒУШЫЛЫҚ АЙМАҒЫ",
        "total_label": "БАРЛЫҒЫ",
        "hours_unit": "сағ",
        "mins_unit": "мин",
        "secs_unit": "сек",
        "folder_link": "🔗 Папкаға сілтеме: {}",
        "files_saved_label": "Файлдар сәтті сақталды:"
    }
}

# ==========================================
# ==========================================
# УПРАВЛЕНИЕ ЯЗЫКОМ (LANGUAGE HANDLING)
# ==========================================
# Файл конфигурации для хранения выбранного языка
CONFIG_FILE = os.path.join(BASE_DIR, "app_config.json")

def get_current_language():
    """
    Получает текущий язык приложения из файла конфигурации.
    
    Returns:
        str: Код языка ('ru', 'en', 'kk'). По умолчанию 'ru'.
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                lang = config.get("language", "ru")
                if lang in LANGUAGES:
                    return lang
        except Exception as e:
            print(f"Ошибка чтения конфигурации языка: {e}")
            pass
    return "ru"

def save_language(lang):
    """
    Сохраняет выбранный язык в файл конфигурации.
    
    Args:
        lang (str): Код языка для сохранения.
    """
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
             print(f"Ошибка чтения конфигурации перед сохранением: {e}")
             pass
    config["language"] = lang
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения языка: {e}")
# ==========================================
# THEME MANAGER
# ==========================================
# ==========================================
# МЕНЕДЖЕР ТЕМ (THEME MANAGER)
# ==========================================
class ThemeManager:
    """
    Класс для управления темами оформления приложения (Светлая/Темная).
    
    Отвечает за:
    - Хранение цветовых схем.
    - Загрузку и сохранение настроек темы.
    - Применение стилей к виджетам Tkinter и TTK.
    """
    
    # Определение цветовых палитр для тем
    THEMES = {
        "light": {
            "bg": "#f0f2f5",         # Основной фон
            "fg": "#333333",         # Основной текст
            "card_bg": "#ffffff",    # Фон карточек/блоков
            "header_bg": "#ffffff",  # Фон заголовка
            "header_fg": "#2c3e50",  # Текст заголовка
            "accent": "#4a90e2",     # Акцентный цвет (синий)
            "accent_hover": "#357abd", # Акцентный цвет при наведении
            "success": "#2ecc71",    # Цвет успеха (зеленый)
            "warning": "#f1c40f",    # Цвет предупреждения (желтый)
            "error": "#e74c3c",      # Цвет ошибки (красный)
            "border": "#dce1e6",     # Цвет границ
            "input_bg": "#ffffff",   # Фон полей ввода
            "input_fg": "#333333",   # Текст полей ввода
            "gray": "#95a5a6",       # Серый цвет (для второстепенного текста)
            "canvas_bg": "#e0e0e0"   # Фон холста
        },
        "dark": {
            "bg": "#1e1e1e",
            "fg": "#e0e0e0",
            "card_bg": "#2d2d2d",
            "header_bg": "#252526",
            "header_fg": "#ffffff",
            "accent": "#4a90e2",
            "accent_hover": "#357abd",
            "success": "#27ae60",
            "warning": "#f39c12",
            "error": "#c0392b",
            "border": "#3e3e42",
            "input_bg": "#3c3c3c",
            "input_fg": "#e0e0e0",
            "gray": "#7f8c8d",
            "canvas_bg": "#121212"
        }
    }

    def __init__(self, config_file=CONFIG_FILE):
        """
        Инициализация менеджера тем.
        
        Args:
            config_file (str): Путь к файлу конфигурации.
        """
        self.config_file = config_file
        self.current_theme = "light" # Тема по умолчанию
        self.load_config()
        self.style = None # Объект стилей ttk

    def load_config(self):
        """Загружает текущую тему из файла конфигурации."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.current_theme = config.get("theme", "light")
            except Exception as e:
                print(f"Ошибка загрузки темы: {e}")

    def save_config(self):
        """Сохраняет текущую тему в файл конфигурации."""
        config = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except:
                pass
        config["theme"] = self.current_theme
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
             print(f"Ошибка сохранения темы: {e}")

    def toggle_theme(self):
        """
        Переключает тему между 'light' и 'dark'.
        
        Returns:
            str: Новое название темы.
        """
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.save_config()
        return self.current_theme

    @property
    def colors(self):
        """Возвращает словарь цветов для текущей темы."""
        return self.THEMES[self.current_theme]

    def setup_styles(self, root):
        """
        Настраивает стили TTK для всего приложения.
        
        Args:
            root (tk.Tk): Корневое окно приложения.
        """
        self.style = ttk.Style(root)
        self.style.theme_use('clam') # Используем 'clam' как базу для лучшей кастомизации
        
        c = self.colors

        # Configure Main Elements
        self.style.configure(".", 
            background=c["bg"], 
            foreground=c["fg"], 
            font=("Segoe UI", 10),
            borderwidth=0
        )
        
        # TFrame
        self.style.configure("TFrame", background=c["bg"])
        self.style.configure("Card.TFrame", background=c["card_bg"], relief="flat")
        
        # TLabel/LabelFrame
        self.style.configure("TLabel", background=c["bg"], foreground=c["fg"], font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), foreground=c["header_fg"], background=c["bg"])
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 12, "bold"), foreground=c["header_fg"], background=c["bg"])
        self.style.configure("Status.TLabel", font=("Segoe UI", 9), foreground=c["gray"])
        self.style.configure("TLabelframe", background=c["bg"], foreground=c["header_fg"], relief="flat", borderwidth=1)
        self.style.configure("TLabelframe.Label", background=c["bg"], foreground=c["header_fg"], font=("Segoe UI", 10, "bold"))

        # Card Style for Frames to simulate cards
        self.style.configure("Card.TLabelframe", background=c["card_bg"], foreground=c["header_fg"], relief="flat", borderwidth=1)
        self.style.configure("Card.TLabelframe.Label", background=c["card_bg"], foreground=c["header_fg"], font=("Segoe UI", 10, "bold"))
        
        # TButton
        self.style.configure("TButton", 
            background=c["accent"], 
            foreground="#ffffff", 
            borderwidth=0, 
            padding=(15, 8), 
            font=("Segoe UI", 10, "bold")
        )
        self.style.map("TButton", 
            background=[('active', c["accent_hover"]), ('disabled', c["gray"])]
        )
        
        # Secondary Button
        self.style.configure("Secondary.TButton", 
            background=c["card_bg"], 
            foreground=c["fg"], 
            borderwidth=1,
            relief="solid",
            bordercolor=c["border"]
        )
        self.style.map("Secondary.TButton", 
            background=[('active', c["border"])],
            foreground=[('active', c["fg"])]
        )

        # TEntry / TSpinbox
        self.style.configure("TEntry", 
            fieldbackground=c["input_bg"], 
            foreground=c["input_fg"], 
            padding=5,
            borderwidth=1,
            relief="solid",
            bordercolor=c["border"]
        )
        self.style.configure("TSpinbox", 
            fieldbackground=c["input_bg"], 
            foreground=c["input_fg"],
            arrowcolor=c["fg"]
        )

        # TCheckbutton / TRadiobutton
        self.style.configure("TCheckbutton", background=c["bg"], foreground=c["fg"], font=("Segoe UI", 10))
        self.style.configure("Card.TCheckbutton", background=c["card_bg"], foreground=c["fg"], font=("Segoe UI", 10))
        self.style.configure("TRadiobutton", background=c["bg"], foreground=c["fg"], font=("Segoe UI", 10))
        self.style.map("TCheckbutton", background=[('active', c["bg"])])
        self.style.map("Card.TCheckbutton", background=[('active', c["card_bg"])])
        self.style.map("TRadiobutton", background=[('active', c["bg"])])

        # TNotebook
        self.style.configure("TNotebook", background=c["bg"], borderwidth=0)
        self.style.configure("TNotebook.Tab", 
            background=c["card_bg"], 
            foreground=c["fg"], 
            padding=(15, 5),
            borderwidth=0,
            font=("Segoe UI", 10)
        )
        self.style.map("TNotebook.Tab", 
            background=[('selected', c["accent"])], 
            foreground=[('selected', "#ffffff")]
        )

        # Progressbar
        self.style.configure("Horizontal.TProgressbar", background=c["accent"], troughcolor=c["border"], borderwidth=0)

        # Configure Root Window
        root.configure(bg=c["bg"])
        self._recursive_style(root, c)

    def apply_to_dialog(self, dialog_window):
        """
        Применяет тему к диалоговому окну и его дочерним элементам.
        
        Это необходимо для стандартных Tkinter виджетов, которые не используют стили TTK.
        
        Args:
            dialog_window (tk.Toplevel): Окно диалога.
        """
        c = self.colors
        dialog_window.configure(bg=c["bg"])
        # Рекурсивно обновляем стили для виджетов
        for widget in dialog_window.winfo_children():
            self._recursive_style(widget, c)

    def _recursive_style(self, widget, colors):
        """
        Внутренний метод для рекурсивного применения цветов к виджетам.
        Особенно важно для Text, Canvas, Listbox, которые не поддерживают ttk стили.
        """
        try:
            widget_type = widget.winfo_class()
            
            # Настройка стандартных текстовых полей и холстов
            if widget_type in ("Text", "Canvas", "Listbox"):
                widget.configure(bg=colors["input_bg"], fg=colors["input_fg"], highlightthickness=0)
            elif widget_type == 'Toplevel':
                widget.configure(bg=colors["bg"])
            elif widget_type == 'Tk':
                widget.configure(bg=colors["bg"])
            elif widget_type == 'Label': # Стандартный tk label (если используется где-то)
                 widget.configure(bg=colors["bg"], fg=colors["fg"])

            # Обработка Frame, если они не TTK
            if widget_type == 'Frame':
                widget.configure(bg=colors["bg"])
            
            # Добавляем поддержку буфера обмена (ПКМ) к текстовым полям
            if widget_type in ("Text", "Entry", "TEntry", "TSpinbox", "Spinbox"):
                self.add_clipboard_menu(widget)
        except Exception as e:
            # Игнорируем ошибки конфигурации для специфических виджетов
            pass
            
        # Рекурсивный вызов для всех дочерних элементов
        for child in widget.winfo_children():
            self._recursive_style(child, colors)

    def add_clipboard_menu(self, widget):
        """
        Добавляет контекстное меню (ПКМ) с опциями Копировать/Вставить/Вырезать.
        
        Args:
            widget: Виджет, к которому привязывается меню.
        """
        lang = get_current_language()
        texts = LANGUAGES.get(lang, LANGUAGES["ru"])
        
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label=texts["copy"], command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label=texts["paste"], command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_command(label=texts["cut"], command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_separator()
        menu.add_command(label=texts["select_all"], command=lambda: self.select_all(widget))
        
        def show_menu(event):
            """Показывает меню при клике ПКМ"""
            menu.tk_popup(event.x_root, event.y_root)
            return "break"
            
        widget.bind("<Button-3>", show_menu)

    def select_all(self, widget):
        """
        Выделяет весь текст в виджете.
        
        Args:
            widget: Текстовый виджет.
        """
        widget.focus_set()
        if hasattr(widget, 'tag_add'): # Для виджета Text
            widget.tag_add(tk.SEL, "1.0", tk.END)
            widget.mark_set(tk.INSERT, "1.0")
            widget.see(tk.INSERT)
        elif hasattr(widget, 'selection_range'): # Для виджета Entry/Spinbox
            widget.selection_range(0, 'end')
        return "break"

def center_window(window):
    """
    Центрирует окно на экране.
    Должно вызываться после того, как все виджеты добавлены и update_idletasks() выполнен.
    """
    window.update_idletasks()
    width = window.winfo_reqwidth()
    height = window.winfo_reqheight()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")


class GoogleDriveUploader:
    """
    Класс для взаимодействия с Google Drive API.
    
    Обязанности:
    - Аутентификация пользователя через OAuth 2.0.
    - Управление токенами доступа (сохранение/обновление).
    - Загрузка файлов и создание папок.
    - Шифрование локального файла конфигурации (для защиты ID папки и настроек).
    """
    def __init__(self, parent_app):
        """
        Инициализация загрузчика Google Drive.
        
        Args:
            parent_app: Ссылка на основной класс приложения (для доступа к ключам шифрования).
        """
        self.parent_app = parent_app
        self.creds = None   # Объект credentials Google Auth
        self.service = None # Объект сервиса Google Drive API
        self.enabled = False # Флаг: включена ли выгрузка
        self.folder_id = None # ID папки назначения в Google Drive
        self.folder_input = None # Текст, введенный пользователем (ссылка или ID)
        self.config_file = os.path.join(BASE_DIR, "gdrive_config.enc") # Файл для хранения настроек
        self.load_config()

    def load_config(self):
        """
        Загружает конфигурацию Google Drive из зашифрованного файла.
        
        Использует ключ Fernet из NotificationSystem основного приложения для дешифровки.
        Это защищает данные конфигурации от простого просмотра.
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'rb') as f:
                    encrypted_data = f.read()
                    # Проверяем наличие ключа шифрования в системе уведомлений
                    if hasattr(self.parent_app.notification_system, 'fernet'):
                        decrypted_data = self.parent_app.notification_system.fernet.decrypt(encrypted_data).decode()
                        config = json.loads(decrypted_data)
                        self.enabled = config.get('enabled', False)
                        self.folder_id = config.get('folder_id', None)
                        self.folder_input = config.get('folder_input', self.folder_id)
            except Exception as e:
                print(f"Error loading Google Drive config: {e}")

    def save_config(self):
        """
        Сохраняет конфигурацию Google Drive в зашифрованный файл.
        """
        config = {
            'enabled': self.enabled,
            'folder_id': self.folder_id,
            'folder_input': self.folder_input
        }
        if hasattr(self.parent_app.notification_system, 'fernet'):
            # Шифруем данные перед записью
            encrypted_data = self.parent_app.notification_system.fernet.encrypt(json.dumps(config).encode())
            with open(self.config_file, 'wb') as f:
                f.write(encrypted_data)
            # Устанавливаем права доступа (чтение/запись только для владельца) для безопасности (Linux/Mac)
            # На Windows это имеет ограниченный эффект, но полезно для кроссплатформенности
            try:
                os.chmod(self.config_file, 0o600)
            except:
                pass

    def authenticate(self):
        """
        Выполняет процесс аутентификации OAuth 2.0.
        
        1. Проверяет наличие сохраненного токена (token.pickle).
        2. Если токен истек, пытается обновить его.
        3. Если токена нет, запускает локальный сервер для входа через браузер.
        4. Сохраняет новый токен.
        
        Returns:
            bool: True, если аутентификация прошла успешно, иначе False.
        """
        # Попытка загрузить сохраненный токен
        if os.path.exists(os.path.join(BASE_DIR, 'token.pickle')):
            with open(os.path.join(BASE_DIR, 'token.pickle'), 'rb') as token:
                self.creds = pickle.load(token)
        
        # Если токена нет или он невалиден
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                # Обновление истекшего токена
                self.creds.refresh(Request())
            else:
                # Полный процесс логина
                if not os.path.exists(os.path.join(BASE_DIR, 'credentials.json')):
                    messagebox.showerror("Ошибка Google Drive",
                                         "Файл credentials.json не найден!\n"
                                         "Чтобы использовать Google Drive:\n"
                                         "1. Перейдите в Google Cloud Console\n"
                                         "2. Создайте проект и включите Drive API\n"
                                         "3. Создайте OAuth 2.0 credentials\n"
                                         "4. Скачайте файл credentials.json в папку с программой")
                    return False
                
                # Инициализация потока OAuth
                flow = InstalledAppFlow.from_client_secrets_file(
                    os.path.join(BASE_DIR, 'credentials.json'), SCOPES)
                # Запуск локального сервера для получения callback от Google
                self.creds = flow.run_local_server(port=0)
            
            # Сохранение токена для будущих запусков
            with open(os.path.join(BASE_DIR, 'token.pickle'), 'wb') as token:
                pickle.dump(self.creds, token)
        
        try:
            # Создание клиента API
            self.service = build('drive', 'v3', credentials=self.creds)
            return True
        except Exception as e:
            print(f"Authentication error: {e}")
            return False

    def create_folder(self, folder_name):
        """
        Создает новую папку в корне Google Drive.
        
        Args:
            folder_name (str): Имя создаваемой папки.
            
        Returns:
            str: ID созданной папки или None в случае ошибки.
        """
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder' # Специальный MIME тип для папок
            }
            folder = self.service.files().create(body=file_metadata, fields='id').execute()
            return folder.get('id')
        except Exception as e:
            print(f"Error creating folder: {e}")
            return None

    def upload_file(self, file_path, folder_id=None):
        """
        Загружает файл в указанную папку на Google Drive.
        Поддерживает докачку (resumable upload).
        
        Args:
            file_path (str): Локальный путь к файлу.
            folder_id (str): ID папки назначения (опционально, иначе используется сохраненный ID).
            
        Returns:
            tuple: (success (bool), file_id_or_error (str))
        """
        if not self.enabled or not self.service:
            return False, "Google Drive не активирован"
        try:
            file_name = os.path.basename(file_path)
            file_metadata = {'name': file_name}
            
            # Установка родительской папки
            if folder_id:
                file_metadata['parents'] = [folder_id]
            elif self.folder_id:
                file_metadata['parents'] = [self.folder_id]
                
            # Создание медиа-объекта для загрузки
            media = MediaFileUpload(file_path, resumable=True)
            
            # Выполнение запроса
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            return True, file.get('id')
        except Exception as e:
            return False, str(e)

    def get_folder_url(self, folder_id):
        """
        Генерирует веб-ссылку на папку Google Drive.
        
        Args:
            folder_id (str): ID папки.
            
        Returns:
            str: URL или None.
        """
        if not folder_id:
            return None
        return f"https://drive.google.com/drive/folders/{folder_id}"


# ==========================================
# ДИАЛОГОВОЕ ОКНО НАСТРОЕК GOOGLE DRIVE (GUI)
# ==========================================
class GoogleDriveSettingsDialog:
    """
    Диалоговое окно для настройки интеграции с Google Drive.
    
    Позволяет пользователю:
    - Пройти авторизацию Google OAuth.
    - Создать специальную папку для загрузок.
    - Выбрать существующую папку по ID или ссылке.
    - Включить/выключить автоматическую загрузку отчетов.
    """
    def __init__(self, parent, gdrive_uploader, lang="ru"):
        self.parent = parent
        self.gdrive = gdrive_uploader
        self.lang = lang
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(LANGUAGES[lang]["gdrive_config"])
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.create_widgets()
        self.load_current_settings()
        center_window(self.dialog)

    def create_widgets(self):
        """Создает элементы интерфейса диалога (виджеты)."""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Фрейм статуса подключения
        status_frame = ttk.LabelFrame(main_frame, text=LANGUAGES[self.lang]["connection_status"])
        status_frame.pack(fill="x", padx=5, pady=5)
        self.status_label = ttk.Label(status_frame, text=LANGUAGES[self.lang]["not_connected"], foreground="red")
        self.status_label.pack(pady=5)
        self.folder_label = ttk.Label(status_frame, text=LANGUAGES[self.lang]["folder_not_selected"])
        self.folder_label.pack(pady=2)

        # Кнопки действий (Авторизация, Создать папку, Открыть папку)
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill="x", padx=5, pady=10)
        ttk.Button(controls_frame, text=LANGUAGES[self.lang]["auth_btn"], command=self.authenticate, style="Secondary.TButton").pack(side="left", padx=5)
        ttk.Button(controls_frame, text=LANGUAGES[self.lang]["create_folder_btn"], command=self.create_folder, style="Secondary.TButton").pack(side="left", padx=5)
        ttk.Button(controls_frame, text=LANGUAGES[self.lang]["open_folder_btn"], command=self.open_folder, style="Secondary.TButton").pack(side="left", padx=5)

        # Фрейм настроек выгрузки
        settings_frame = ttk.LabelFrame(main_frame, text=LANGUAGES[self.lang]["upload_settings"])
        settings_frame.pack(fill="x", padx=5, pady=10)
        self.enabled_var = tk.BooleanVar(value=self.gdrive.enabled)
        ttk.Checkbutton(settings_frame, text=LANGUAGES[self.lang]["auto_upload"],
                        variable=self.enabled_var).pack(anchor="w", padx=10, pady=5)
        ttk.Label(settings_frame, text=LANGUAGES[self.lang]["folder_id_label"]).pack(anchor="w", padx=10, pady=5)
        self.folder_id_entry = ttk.Entry(settings_frame, width=50)
        self.folder_id_entry.pack(fill="x", padx=10, pady=5)

        # Фрейм инструкций
        instructions_frame = ttk.LabelFrame(main_frame, text=LANGUAGES[self.lang]["instructions_title"])
        instructions_frame.pack(fill="both", expand=True, padx=5, pady=10)
        instructions = LANGUAGES[self.lang]["instructions_text"]
        ttk.Label(instructions_frame, text=instructions, justify="left",
                  wraplength=450).pack(padx=10, pady=10)

        # Кнопки управления (Сохранить, Отмена)
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["save"], command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["cancel"], command=self.dialog.destroy).pack(side="right", padx=5)

    def load_current_settings(self):
        self.enabled_var.set(self.gdrive.enabled)
        # Show folder_input if available, otherwise fallback to folder_id
        input_value = self.gdrive.folder_input if self.gdrive.folder_input else (self.gdrive.folder_id if self.gdrive.folder_id else "")
        self.folder_id_entry.insert(0, input_value)
        self.update_status()

    def update_status(self):
        if hasattr(self.gdrive, 'service') and self.gdrive.service:
            self.status_label.config(text=LANGUAGES[self.lang]["connected"], foreground="green")
        else:
            self.status_label.config(text=LANGUAGES[self.lang]["not_connected"], foreground="red")
        if self.gdrive.folder_id:
            self.folder_label.config(text=LANGUAGES[self.lang]["folder"].format(self.gdrive.folder_id))
        else:
            self.folder_label.config(text=LANGUAGES[self.lang]["folder_not_selected"])

    def authenticate(self):
        if self.gdrive.authenticate():
            self.update_status()
            messagebox.showinfo(LANGUAGES[self.lang]["success"], LANGUAGES[self.lang]["auth_success"])
        else:
            self.update_status()
            messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["auth_fail"])

    def create_folder(self):
        if not hasattr(self.gdrive, 'service') or not self.gdrive.service:
            messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["auth_common_error"])
            return
        folder_name = simpledialog.askstring(LANGUAGES[self.lang]["folder_create_title"],
                                             LANGUAGES[self.lang]["folder_name_prompt"],
                                             parent=self.dialog)
        if folder_name:
            folder_id = self.gdrive.create_folder(folder_name)
            if folder_id:
                self.folder_id_entry.delete(0, tk.END)
                self.folder_id_entry.insert(0, folder_id)
                messagebox.showinfo(LANGUAGES[self.lang]["success"], LANGUAGES[self.lang]["folder_create_success"].format(folder_name, folder_id))
                self.update_status()
            else:
                messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["folder_create_fail"])

    def open_folder(self):
        folder_input = self.folder_id_entry.get().strip()
        folder_id = self.extract_folder_id(folder_input)
        if not folder_id:
            folder_id = self.gdrive.folder_id
        if folder_id:
            url = self.gdrive.get_folder_url(folder_id)
            if url:
                webbrowser.open(url)
                return
        messagebox.showinfo(LANGUAGES[self.lang]["info_title"], LANGUAGES[self.lang]["folder_select_error"])

    def extract_folder_id(self, link_or_id):
        """
        Извлекает ID папки Google Drive из полной ссылки или возвращает ID как есть.
        Поддерживает форматы:
        - Просто ID (набор символов)
        - Ссылка: .../folders/ID...
        - Ссылка: ...?id=ID...
        
        Args:
            link_or_id (str): Ввод пользователя.
            
        Returns:
            str: Извлеченный ID или исходная строка.
        """
        if not link_or_id:
            return None
        link_or_id = link_or_id.strip()
        
        # Если это ссылка на Google Drive
        if "drive.google.com" in link_or_id:
            import re
            # Паттерн 1: Стандартный формат /folders/ID
            match = re.search(r'folders/([a-zA-Z0-9-_]{25,})', link_or_id)
            if match:
                return match.group(1)
            # Паттерн 2: Формат с параметром ?id=ID
            match = re.search(r'id=([a-zA-Z0-9-_]{25,})', link_or_id)
            if match:
                return match.group(1)
                
        # Если ссылка не распознана, считаем, что пользователь ввел ID напрямую
        return link_or_id

    def save_settings(self):
        self.gdrive.enabled = self.enabled_var.get()
        folder_input = self.folder_id_entry.get().strip()
        self.gdrive.folder_input = folder_input if folder_input else None
        self.gdrive.folder_id = self.extract_folder_id(folder_input)
        self.gdrive.save_config()
        self.update_status()
        messagebox.showinfo(LANGUAGES[self.lang]["settings_saved"], LANGUAGES[self.lang]["config_saved_gdrive"])
        self.dialog.destroy()

    def show(self):
        self.dialog.wait_window()


# ==========================================
# КЛАСС ЛОГИКИ ЛИНИЙ И ЗОН (без изменений)
# ==========================================
class CounterLine:
    """
    Представляет виртуальную линию или зону для подсчета людей.
    
    Attributes:
        p1 (tuple): Начальная точка линии (x, y).
        p2 (tuple): Конечная точка линии (x, y). Для зоны может быть None.
        type (str): Тип элемента: 'entry' (Вход), 'exit' (Выход), 'control' (Контроль), 'zone' (Зона интереса).
        points (list): Список точек [(x,y), ...] образующих полигон (только для типа 'zone').
        min_x, max_x, min_y, max_y: Границы (Bounding Box) для быстрой проверки зон.
        color (tuple): Цвет для отрисовки элемента (B, G, R).
    """
    def __init__(self, p1, p2=None, line_type="control", points=None, zone_id=None):
        self.p1 = p1
        self.p2 = p2
        self.type = line_type
        self.points = points if points is not None else []  # Для зон интереса
        self.zone_id = zone_id # Уникальный ID для зон
        self.count = 0        # Индивидуальный счетчик для зоны
        self.color = (0, 255, 255)
        self.update_color()
        if self.type == "zone" and self.points:
            self.calculate_bounds()

    def calculate_bounds(self):
        """
        Вычисляет прямоугольные границы (Bounding Box) полигона для оптимизации проверки попадания точки.
        """
        if self.type == "zone" and self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.min_x, self.max_x = min(xs), max(xs)
            self.min_y, self.max_y = min(ys), max(ys)

    def update_color(self):
        """Обновляет цвет линии в зависимости от её типа."""
        if self.type == "entry":
            self.color = (0, 255, 0)    # Зеленый
        elif self.type == "exit":
            self.color = (0, 0, 255)   # Красный
        elif self.type == "zone":
            # Палитра цветов для зон (BGR)
            palette = [
                (255, 0, 255),   # Фиолетовый
                (0, 165, 255),   # Оранжевый
                (255, 255, 0),   # Бирюзовый
                (0, 255, 255),   # Желтый
                (255, 0, 0),     # Синий
                (0, 255, 0),     # Зеленый
                (128, 0, 128),   # Темно-фиолетовый
                (0, 128, 255)    # Светло-оранжевый
            ]
            if self.zone_id:
                self.color = palette[(self.zone_id - 1) % len(palette)]
            else:
                self.color = (255, 0, 255) # Фиолетовый по умолчанию
        else:
            self.color = (0, 255, 255)                      # Желтый (контроль)

    def is_intersecting(self, old_pos, new_pos):
        """
        Проверяет, пересекает ли вектор движения (old_pos -> new_pos) линию подсчета.
        
        Использует алгоритм CCW (Counter-Clockwise) для определения пересечения двух отрезков.
        
        Args:
            old_pos (tuple): Предыдущая координата центра объекта (x, y).
            new_pos (tuple): Текущая координата центра объекта (x, y).
            
        Returns:
            bool: True, если произошло пересечение, иначе False.
        """
        if self.type == "zone":
            return self.point_in_polygon(new_pos)

        # Вложенная функция для проверки ориентации тройки точек
        def ccw(A, B, C):
            # Возвращает > 0 если A-B-C против часовой стрелки, < 0 если по часовой, 0 если коллинеарны
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B, C, D = self.p1, self.p2, old_pos, new_pos
        # Пересечение происходит, если ориентации троек точек различаются для обоих отрезков
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def point_in_polygon(self, point):
        """
        Проверяет, находится ли точка внутри зоны полигона.
        
        Использует алгоритм Ray Casting (трассировка луча): выпускается луч из точки
        в произвольном направлении (здесь горизонтально вправо). Если луч пересекает
        границы полигона нечетное количество раз, точка находится внутри.
        
        Args:
            point (tuple): Координаты точки (x, y).
            
        Returns:
            bool: True, если точка внутри полигона.
        """
        if self.type != "zone" or not self.points:
            return False
        x, y = point
        inside = False
        n = len(self.points)
        
        # 1. Быстрая проверка по Bounding Box (отсекает большинство точек снаружи)
        if not (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            return False
            
        # 2. Детальная проверка Ray Casting
        p1 = self.points[0]
        for i in range(1, n + 1):
            p2 = self.points[i % n]
            # Проверяем пересечение горизонтального луча с ребром полигона
            if y > min(p1[1], p2[1]) and y <= max(p1[1], p2[1]) and x <= max(p1[0], p2[0]):
                if p1[1] != p2[1]:
                    xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                # Меняем статус inside при каждом пересечении
                if p1[0] == p2[0] or x <= xinters:
                    inside = not inside
            p1 = p2
        return inside


# ==========================================
# КЛАСС СИСТЕМЫ ОПОВЕЩЕНИЙ (без изменений)
# ==========================================
class NotificationSystem:
    """
    Система управления уведомлениями (Email).
    
    Обязанности:
    - Хранение пороговых значений (thresholds) для различных счетчиков.
    - Управление списком контактов для оповещений.
    - Отправка Email при превышении порогов (с защитой от спама через кулдаун).
    - Шифрование настроек SMTP и паролей.
    """
    def __init__(self):
        self.config_file = "notification_config.enc" # Файл зашифрованной конфигурации
        self.key_file = "secret.key" # Файл ключа шифрования
        self.email_enabled = False
        # Пороговые значения по умолчанию
        self.thresholds = {
            "entered": 100,
            "exited": 100,
            "control": 50,
            "zone": 20,
            "total": 100
        }
        self.contacts = {
            "emails": []
        }
        self.email_settings = {} # Настройки SMTP
        
        # Словарь для хранения времени последней отправки уведомления для каждого типа
        self.last_notification_time = {}
        self.cooldown = 300  # Период "остывания" (в секундах) между уведомлениями одного типа (5 мин)
        
        self.fernet = self.load_or_generate_key()
        self.load_config()

    def load_or_generate_key(self):
        """
        Загружает ключ шифрования (Fernet) или генерирует новый, если он отсутствует.
        Это гарантирует, что зашифрованные конфигурации могут быть прочитаны только на этом компьютере.
        """
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Установка прав доступа (только чтение владельцем)
            try:
                os.chmod(self.key_file, 0o600)
            except:
                pass
        return Fernet(key)

    def encrypt_data(self, data):
        """Шифрует словарь данных в строку."""
        return self.fernet.encrypt(json.dumps(data).encode()).decode()

    def decrypt_data(self, encrypted_data):
        """Расшифровывает строку данных в словарь."""
        try:
            return json.loads(self.fernet.decrypt(encrypted_data.encode()).decode())
        except:
            return None

    def load_config(self):
        """Загружает конфигурацию из зашифрованного файла."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    encrypted_config = f.read()
                    config = self.decrypt_data(encrypted_config)
                    if config:
                        self.email_enabled = config.get('email_enabled', False)
                        self.thresholds = config.get('thresholds', self.thresholds)
                        self.contacts = config.get('contacts', self.contacts)
                        self.email_settings = config.get('email_settings', {})
            except Exception as e:
                print(f"Error loading notification config: {e}")

    def save_config(self):
        """Сохраняет текущую конфигурацию в зашифрованный файл."""
        thresholds_to_save = self.thresholds
        config = {
            'email_enabled': self.email_enabled,
            'thresholds': thresholds_to_save,
            'contacts': self.contacts,
            'email_settings': self.email_settings,
        }
        encrypted_config = self.encrypt_data(config)
        with open(self.config_file, 'w') as f:
            f.write(encrypted_config)
        try:
            os.chmod(self.config_file, 0o600)
        except:
            pass
    def test_email_connection(self, settings):
        """Проверяет настройки SMTP путем попытки входа на сервер."""
        lang = get_current_language()
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(settings['smtp_server'], int(settings['smtp_port']), timeout=10) as server:
                server.starttls(context=context)
                server.login(settings['email'], settings['password'])
            return True, LANGUAGES[lang]["connection_success"]
        except Exception as e:
            return False, str(e)

    def send_email_notification(self, subject, message, counts):
        """
        Отправляет форматированное HTML-письмо с уведомлением.
        
        Args:
            subject (str): Тема письма.
            message (str): Текст сообщения (причина уведомления).
            counts (dict): Текущая статистика счетчиков для включения в отчет.
        """
        if not self.email_enabled or not self.email_settings:
            return False
        
        lang = get_current_language()
        L = LANGUAGES[lang]
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_settings['email']
            msg['To'] = ', '.join(self.contacts['emails'])
            msg['Subject'] = subject
            
            # Формирование HTML тела письма
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
            <h2 style="color: #2c3e50;">{L['email_notif_header']}</h2>
            <p><strong>{L['email_notif_time']}</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3>{L['email_notif_stats']}</h3>
            <ul style="list-style: none; padding: 0;">
            <li style="color: green; margin: 5px 0;">📥 {L['entry']}: {counts['entered']} ({L['threshold_simple']}: {self.thresholds['entered']})</li>
            <li style="color: red; margin: 5px 0;">📤 {L['exit']}: {counts['exited']} ({L['threshold_simple']}: {self.thresholds['exited']})</li>
            <li style="color: #ffcc00; margin: 5px 0;">🔍 {L['control']}: {counts['control']} ({L['threshold_simple']}: {self.thresholds['control']})</li>
            <li style="color: purple; margin: 5px 0;">🎯 {L['interest_zone']}: {counts['zone']} ({L['threshold_simple']}: {self.thresholds['zone']})</li>
            <li style="color: blue; margin: 5px 0;">👥 {L['total']}: {counts['total']} ({L['threshold_simple']}: {self.thresholds['total']})</li>
            </ul>
            </div>
            <p>{message}</p>
            <hr>
            <p style="font-size: 12px; color: #666;">{L['email_notif_footer']}</p>
            </body>
            </html>
            """
            msg.attach(MIMEText(html_body, 'html'))
            
            # Отправка
            context = ssl.create_default_context()
            with smtplib.SMTP(self.email_settings['smtp_server'], int(self.email_settings['smtp_port'])) as server:
                server.starttls(context=context)
                server.login(self.email_settings['email'], self.email_settings['password'])
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Ошибка отправки email: {e}")
            return False

    def check_thresholds_and_notify(self, counts, force=False):
        """
        Проверяет все счетчики на превышение порогов и инициирует отправку уведомлений.
        
        Запускает отправку письма В ОТДЕЛЬНОМ ПОТОКЕ, чтобы не блокировать основной поток обработки видео.
        
        Args:
            counts (dict): Текущие значения счетчиков.
            force (bool): Если True, игнорирует кулдаун (время ожидания).
            
        Returns:
            bool: True, если хотя бы одно уведомление было инициировано.
        """
        notifications_sent = False
        lang = get_current_language()
        for counter_type, current_count in counts.items():
            threshold = self.thresholds.get(counter_type, float('inf'))
            last_time = self.last_notification_time.get(counter_type, 0)
            current_time = time.time()
            
            # Условие: Превышен порог И (прошел кулдаун ИЛИ принудительная отправка)
            if current_count >= threshold and (force or (current_time - last_time) > self.cooldown):
                counter_label = LANGUAGES[lang].get(f"{counter_type}_label", counter_type.upper())
                subject = LANGUAGES[lang]["notif_subject_threshold"].format(counter_label, current_count)
                message = LANGUAGES[lang]["notif_msg_threshold"].format(counter_label, current_count, threshold)
                
                if self.email_enabled:
                    # Запуск отправки в фоне
                    email_thread = threading.Thread(
                        target=self.send_email_notification,
                        args=(subject, message, counts),
                        daemon=True
                    )
                    email_thread.start()

                self.last_notification_time[counter_type] = current_time
                notifications_sent = True
        return notifications_sent


# ==========================================
# ДИАЛОГОВОЕ ОКНО НАСТРОЕК УВЕДОМЛЕНИЙ (GUI)
# ==========================================
class NotificationSettingsDialog:
    """
    Диалоговое окно для настройки системы уведомлений.
    
    Позволяет пользователю:
    - Настраивать параметры SMTP для отправки Email.
    - Устанавливать пороговые значения (thresholds) для различных счетчиков.
    - Управлять списком Email-адресов для получения оповещений.
    """
    def __init__(self, parent, notification_system, lang="ru"):
        self.parent = parent
        self.notif_sys = notification_system
        self.lang = lang
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(LANGUAGES[lang]["notif_config"])
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.create_widgets()
        self.load_current_settings()
        center_window(self.dialog)

    def create_widgets(self):
        """Создает элементы интерфейса диалога (виджеты)."""
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Вкладка Email настроек
        email_frame = ttk.Frame(notebook)
        notebook.add(email_frame, text="📧 Email")
        self.create_email_tab(email_frame)

        # Вкладка порогов и контактов
        contacts_frame = ttk.Frame(notebook)
        notebook.add(contacts_frame, text=LANGUAGES[self.lang]["thresholds_contacts_tab"])
        self.create_thresholds_tab(contacts_frame)
        
        # Кнопки управления (Сохранить, Тест, Отмена)
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["save"], command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["test_email"], command=self.test_email, style="Secondary.TButton").pack(side="left", padx=5)
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["cancel"], command=self.dialog.destroy).pack(side="right", padx=5)

    def create_email_tab(self, frame):
        """Заполняет вкладку настроек SMTP."""
        ttk.Label(frame, text=LANGUAGES[self.lang]["smtp_server_label"]).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.smtp_server = ttk.Entry(frame, width=40)
        self.smtp_server.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text=LANGUAGES[self.lang]["port_label"]).grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.smtp_port = ttk.Entry(frame, width=10)
        self.smtp_port.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(frame, text=LANGUAGES[self.lang]["email_label"]).grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.email_entry = ttk.Entry(frame, width=40)
        self.email_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text=LANGUAGES[self.lang]["password_label"]).grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.password_entry = ttk.Entry(frame, width=40, show="*")
        self.password_entry.grid(row=3, column=1, padx=5, pady=5)
        
        self.show_password_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text=LANGUAGES[self.lang]["show_password"], variable=self.show_password_var,
                        command=self.toggle_password_visibility).grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        self.email_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text=LANGUAGES[self.lang]["enable_email"], 
                variable=self.email_enabled_var).grid(row=5, column=0, columnspan=2, padx=5, pady=10, sticky="w")



    def create_thresholds_tab(self, frame):
        """Заполняет вкладку пороговых значений и контактов."""
        # Секция порогов
        thresholds_frame = ttk.LabelFrame(frame, text=LANGUAGES[self.lang]["thresholds_tab"])
        thresholds_frame.pack(fill="x", padx=10, pady=10)
        self.threshold_vars = {}
        row = 0
        for counter_type, default_value in [('entered', 100), ('exited', 100), ('control', 50), ('zone', 20), ('total', 100)]:
            counter_label = LANGUAGES[self.lang].get(f"{counter_type}_label", counter_type.upper())
            ttk.Label(thresholds_frame, text=f"{counter_label}{LANGUAGES[self.lang]['threshold_label_suffix']}").grid(row=row, column=0, padx=5, pady=5, sticky="e")
            var = tk.IntVar(value=default_value)
            self.threshold_vars[counter_type] = var
            ttk.Spinbox(thresholds_frame, from_=1, to=10000, width=10, textvariable=var).grid(row=row, column=1, padx=5, pady=5, sticky="w")
            row += 1
            
        # Секция контактов (только Email)
        contacts_frame_inner = ttk.LabelFrame(frame, text=LANGUAGES[self.lang]["contacts_frame_title"])
        contacts_frame_inner.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(contacts_frame_inner, text=LANGUAGES[self.lang]["emails_label"]).pack(anchor="w", padx=5, pady=5)
        self.email_contacts = tk.Text(contacts_frame_inner, height=5, width=50)
        self.email_contacts.pack(fill="x", padx=5, pady=5)

    def load_current_settings(self):
        """Загружает текущие настройки из системы уведомлений в виджеты."""
        self.email_enabled_var.set(self.notif_sys.email_enabled)
        if hasattr(self.notif_sys, 'email_settings'):
            self.smtp_server.insert(0, self.notif_sys.email_settings.get('smtp_server', 'smtp.gmail.com'))
            self.smtp_port.insert(0, self.notif_sys.email_settings.get('smtp_port', '587'))
            self.email_entry.insert(0, self.notif_sys.email_settings.get('email', ''))
            self.password_entry.insert(0, self.notif_sys.email_settings.get('password', ''))

        # Загрузка порогов
        for counter_type, var in self.threshold_vars.items():
            var.set(self.notif_sys.thresholds.get(counter_type, var.get()))
        self.email_contacts.insert("1.0", ", ".join(self.notif_sys.contacts.get('emails', [])))



    def toggle_password_visibility(self):
        if self.show_password_var.get():
            self.password_entry.config(show="")

        else:
            self.password_entry.config(show="*")


    def test_email(self):
        settings = {
            'smtp_server': self.smtp_server.get(),
            'smtp_port': self.smtp_port.get(),
            'email': self.email_entry.get(),
            'password': self.password_entry.get()
        }
        success, message = self.notif_sys.test_email_connection(settings)
        if success:
            messagebox.showinfo(LANGUAGES[self.lang]["success"], LANGUAGES[self.lang]["test_email_success"].format(message))
        else:
            messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["connection_error"].format(message))



    def save_settings(self):
        email_settings = {
            'smtp_server': self.smtp_server.get(),
            'smtp_port': self.smtp_port.get(),
            'email': self.email_entry.get(),
            'password': self.password_entry.get()
        }

        # Сохраняем пороги
        thresholds = {k: v for k, v in self.threshold_vars.items()}
        thresholds_final = {k: v.get() for k, v in thresholds.items()}
        email_contacts = [email.strip() for email in self.email_contacts.get("1.0", tk.END).split(",") if email.strip()]
        contacts = {
            'emails': email_contacts
        }
        self.notif_sys.email_enabled = self.email_enabled_var.get()
        self.notif_sys.email_settings = email_settings
        self.notif_sys.thresholds = thresholds_final
        self.notif_sys.contacts = contacts
        self.notif_sys.save_config()
        messagebox.showinfo(LANGUAGES[self.lang]["settings_saved"], LANGUAGES[self.lang]["config_saved"])
        self.dialog.destroy()

    def show(self):
        self.dialog.wait_window()
        return {
            'email_enabled': self.email_enabled_var.get(),
            'email_enabled': self.email_enabled_var.get()
        }


# ==========================================
# ДИАЛОГОВОЕ ОКНО НАСТРОЕК ТЕПЛОВОЙ КАРТЫ (GUI)
# ==========================================
class HeatmapSettingsDialog:
    """
    Диалоговое окно для настройки параметров генерации тепловой карты (Heatmap).
    
    Параметры:
    - Прозрачность: Наложение тепловой карты на исходное видео.
    - Радиус размытия: Размер пятна тепла вокруг человека.
    - Коэффициент затухания: Скорость исчезновения "тепла" с карты со временем.
    """
    def __init__(self, parent, app_instance, lang="ru"):
        self.parent = parent
        self.app = app_instance
        self.lang = lang
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(LANGUAGES[lang]["heatmap_params"])
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.create_widgets()
        self.load_current_settings()
        center_window(self.dialog)

    def create_widgets(self):
        """Создает элементы интерфейса для настройки тепловой карты."""
        frame = ttk.Frame(self.dialog)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Настройка прозрачности (Alpha)
        ttk.Label(frame, text=LANGUAGES[self.lang]["heatmap_transparency"]).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.alpha_var = tk.DoubleVar(value=self.app.heatmap_alpha)
        self.alpha_scale = ttk.Scale(frame, from_=0.0, to=1.0, orient="horizontal", variable=self.alpha_var, command=lambda v: self.update_labels())
        self.alpha_scale.grid(row=0, column=1, padx=5, pady=5)
        self.alpha_label = ttk.Label(frame, text=f"{self.alpha_var.get():.2f}")
        self.alpha_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Настройка размытия (Blur Kernel size)
        ttk.Label(frame, text=LANGUAGES[self.lang]["heatmap_blur"]).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.blur_var = tk.StringVar(value=str(self.app.heatmap_blur_kernel[0]))
        self.blur_spinbox = ttk.Spinbox(frame, from_=1, to=99, increment=2, textvariable=self.blur_var, width=10)
        self.blur_spinbox.grid(row=1, column=1, padx=5, pady=5)
        
        # Настройка затухания (Decay Factor)
        ttk.Label(frame, text=LANGUAGES[self.lang]["heatmap_decay"]).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.decay_var = tk.DoubleVar(value=self.app.heatmap_decay_factor)
        self.decay_scale = ttk.Scale(frame, from_=0.90, to=1.00, orient="horizontal", variable=self.decay_var, command=self.on_decay_change)
        self.decay_scale.grid(row=2, column=1, padx=5, pady=5)
        self.decay_label = ttk.Label(frame, text=f"{self.decay_var.get():.2f}")
        self.decay_label.grid(row=2, column=2, padx=5, pady=5)
        
        # Кнопки сохранения и выхода
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["save"], command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["cancel"], command=self.dialog.destroy).pack(side="right", padx=5)

    def load_current_settings(self):
        self.alpha_var.set(self.app.heatmap_alpha)
        self.blur_var.set(str(self.app.heatmap_blur_kernel[0]))
        self.decay_var.set(self.app.heatmap_decay_factor)
        self.update_labels()

    def save_settings(self):
        try:
            alpha = float(self.alpha_var.get())
            blur_size = int(self.blur_var.get())
            decay = float(self.decay_var.get())
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("Alpha must be between 0.0 and 1.0")
            if blur_size < 1 or blur_size % 2 == 0:
                raise ValueError("Blur size must be an odd positive integer")
            if not (0.90 <= decay <= 1.00):
                raise ValueError("Decay must be between 0.90 and 1.00")
            self.app.heatmap_alpha = alpha
            self.app.heatmap_blur_kernel = (blur_size, blur_size)
            self.app.heatmap_decay_factor = decay
            messagebox.showinfo(LANGUAGES[self.lang]["settings_saved"], LANGUAGES[self.lang]["heatmap_settings_applied"])
            self.dialog.destroy()
        except ValueError as e:
            messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["invalid_value"].format(e))

    def update_labels(self):
        self.alpha_label.config(text=f"{self.alpha_var.get():.2f}")
        self.decay_label.config(text=f"{self.decay_var.get():.2f}")

    def on_decay_change(self, val):
        self.decay_label.config(text=f"{float(val):.2f}")


# ==========================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ С ИНТЕГРАЦИЕЙ GOOGLE DRIVE
# ==========================================
class PeopleCounterPRO:
    """
    Main application class for the AI People Counter.

    This class handles the GUI initialization, video processing, object tracking,
    heatmap generation, and integration with external services like Google Drive and Email.

    Attributes:
        root (tk.Tk): The main Tkinter window.
        lang (str): Current language code ('en', 'ru', 'kk').
        counts (dict): Dictionary storing current counts (entered, exited, control, zone, total).
        tracked_objects (dict): Dictionary storing tracked object states.
        heatmap_enabled (bool): Flag to toggle heatmap generation.
        google_drive (GoogleDriveUploader): Handler for Google Drive operations.
        notification_system (NotificationSystem): System for handling notifications.
    """
    def __init__(self):
        """
        Initialize the People Counter application.
        
        Sets up the main window, loads configuration, initializes variables,
        and starts the GUI loop.
        """
        self.root = tk.Tk() # Initialize Root first for tk.Variables
        self.theme_manager = ThemeManager() # Initialize ThemeManager
        self.lang = get_current_language()

        # Попытка установить REALTIME приоритет процесса для максимальной производительности
        try:
            p = psutil.Process(os.getpid())
            if sys.platform == 'win32':
                # Пытаемся установить REALTIME. Если не удастся (нет прав админа), установим HIGH.
                try:
                    p.nice(psutil.REALTIME_PRIORITY_CLASS)
                except:
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                p.nice(10)  # Для Linux/macOS
            
            # Логируем результат
            current_priority = p.nice()
            if sys.platform == 'win32':
                p_name = "REALTIME" if current_priority == 256 else ("HIGH" if current_priority == 128 else str(current_priority))
                print(f"Process priority set to: {p_name} ({current_priority})")
                if current_priority != 256:
                    print("Note: REALTIME priority requires administrative privileges. Using HIGH instead.")
                
                # Попытка установить высокий приоритет ввода-вывода (I/O)
                try:
                    if hasattr(psutil, 'IOPRIO_HIGH'):
                        p.ionice(psutil.IOPRIO_HIGH)
                        print("I/O priority set to HIGH")
                except:
                    pass
            else:
                p.nice(10)  # Для Linux/macOS
                print(f"Process priority set to: {p.nice()}")
        except Exception as e:
            print(f"Could not set process priority: {e}")

        # Глобальные оптимизации PyTorch для GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
        # Оптимизация многопоточности (использование физических ядер)
        try:
            physical_cores = psutil.cpu_count(logical=False) or 4
            torch.set_num_threads(physical_cores)
            cv2.setNumThreads(physical_cores)
        except:
            pass

        # Загрузка настроек GPU из конфига
        gpu_accel_config = True # По умолчанию включено
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    gpu_accel_config = config.get("gpu_accel", True)
            except:
                pass
        
        self.var_gpu_accel = tk.BooleanVar(value=gpu_accel_config)
        self.var_turbo_mode = tk.BooleanVar(value=False)
        self.init_torch_device()
        
        self.model = YOLO("yolov8n.pt").to(self.device)
        self.lines = []
        self.drawing_line = []
        # Убран 'total' из начального словаря
        self.counts = {"entered": 0, "exited": 0, "control": 0, "zone": 0, "total": 0}
        self.tracked_objects = {}  # now stores (cx, cy, entered_zone)
        self.last_seen_time = {}
        self.current_draw_mode = "control"
        # --- Новые атрибуты для тепловой карты ---
        self.heatmap_enabled = False
        self.save_heatmap = False
        self.heatmap_alpha = 0.5
        self.heatmap_blur_kernel = (15, 15)
        self.heatmap_decay_factor = 0.98
        self.heatmap = None
        self.heatmap_lock = threading.Lock()
        self.video_processing = False
        self.abort_processing = False
        self.markup_confirmed = False
        self.save_csv = True
        self.notification_system = NotificationSystem()
        # Инициализация Google Drive
        self.google_drive = GoogleDriveUploader(self)
        
        # --- FPS and Face Blur ---
        self.face_blur_enabled = False
        self.var_face_blur = tk.BooleanVar(value=False)
        self.fps = 0
        self.last_frame_time = time.time()
        
        # Загрузка каскада для лиц
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
             # Попытка поиска в альтернативных путях или просто игнорирование
             print("Warning: Face cascade not found. Using box-based fallback.")
        
        self.init_gui()
        # ГЛАВНОЕ ИЗМЕНЕНИЕ: обновляем статус сразу после инициализации GUI
        self.update_notification_status()
        self.update_gdrive_status()
        
        # Start the application
        self.root.mainloop()

    def init_gui(self):
        """
        Initialize the Graphical User Interface (GUI).
        
        Sets up the main window, themes, header section, and main content area.
        Configures source selection, timer, system settings, and other UI components.
        """
        self.root.title(LANGUAGES[self.lang]["app_title"])
        self.root.geometry("680x950")
        self.root.state('zoomed') # Open in full screen (maximized)
        
        # Bring to front on startup
        self.root.lift()
        self.root.attributes('-topmost',True)
        self.root.after_idle(self.root.attributes,'-topmost',False)
        
        # Apply Initial Theme
        self.theme_manager.setup_styles(self.root)

        # === Header Section ===
        header_frame = ttk.Frame(self.root, padding=(20, 15))
        header_frame.pack(fill="x")
        
        ttk.Label(header_frame, text=LANGUAGES[self.lang]["app_title"], style="Header.TLabel").pack(side="left")
        
        # Theme Toggle
        theme_btn_text = "🌙" if self.theme_manager.current_theme == "light" else "☀️"
        self.theme_btn = ttk.Button(header_frame, text=theme_btn_text, command=self.toggle_theme_ui, style="Secondary.TButton", width=3)
        self.theme_btn.pack(side="right", padx=5)

        # Language Menu Button (replace system menu)
        lang_mb = ttk.Menubutton(header_frame, text=LANGUAGES[self.lang]["language"], direction='below', style="Secondary.TButton")
        lang_menu = tk.Menu(lang_mb, tearoff=0)
        lang_mb.configure(menu=lang_menu)
        lang_menu.add_command(label="English", command=lambda: self.set_language("en"))
        lang_menu.add_command(label="Русский", command=lambda: self.set_language("ru"))
        lang_menu.add_command(label="Қазақша", command=lambda: self.set_language("kk"))
        lang_mb.pack(side="right", padx=5)

        # === Scrollable Content Area ===
        self.container = ttk.Frame(self.root)
        self.container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.container, highlightthickness=0, background=self.theme_manager.colors["bg"])
        self.scrollbar = ttk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        
        # Frame that will contain all the settings
        self.main_content = ttk.Frame(self.canvas, padding=20)
        
        # Configure the canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.main_content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bindings for resizing and scrolling
        self.main_content.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # === Основная область контента (теперь внутри main_content) ===

        # Конфигурация источника видео (Карточка)
        src_frame = ttk.LabelFrame(self.main_content, text=LANGUAGES[self.lang]["video_source"], style="Card.TLabelframe", padding=15)
        src_frame.pack(fill="x", pady=(0, 15))
        
        self.source_type = tk.StringVar(value="webcam")
        
        # Сетка для выбора источника
        # Веб-камера
        ttk.Radiobutton(src_frame, text=LANGUAGES[self.lang]["webcam"], variable=self.source_type, value="webcam", 
                        command=self.toggle_source, style="Card.TRadiobutton").grid(row=0, column=0, sticky="w", pady=5)
        self.spn_cam_idx = ttk.Spinbox(src_frame, from_=0, to=10, width=5)
        self.spn_cam_idx.set(0)
        self.spn_cam_idx.grid(row=0, column=1, sticky="w", padx=10)

        # IP Камера (RTSP/HTTP)
        ttk.Radiobutton(src_frame, text=LANGUAGES[self.lang]["ip_camera"], variable=self.source_type, value="ip", 
                        command=self.toggle_source, style="Card.TRadiobutton").grid(row=1, column=0, sticky="w", pady=5)
        self.ent_ip_url = ttk.Entry(src_frame, width=40)
        self.ent_ip_url.insert(0, "rtsp://admin:password@192.168.1.100:554/path")
        self.ent_ip_url.config(state="disabled")
        self.ent_ip_url.grid(row=1, column=1, sticky="w", padx=10)

        # Видео файл
        ttk.Radiobutton(src_frame, text=LANGUAGES[self.lang]["video_file"], variable=self.source_type, value="file", 
                        command=self.toggle_source, style="Card.TRadiobutton").grid(row=2, column=0, sticky="w", pady=5)
        frame_file_sel = ttk.Frame(src_frame, style="Card.TFrame")
        frame_file_sel.grid(row=2, column=1, sticky="w", padx=10)
        self.btn_select_file = ttk.Button(frame_file_sel, text=LANGUAGES[self.lang]["select_file"], 
                                         command=self.select_video_file, state="disabled", style="Secondary.TButton")
        self.btn_select_file.pack(side="left")
        self.selected_file_path = tk.StringVar(value="")
        ttk.Label(src_frame, textvariable=self.selected_file_path, wraplength=400, 
                  foreground=self.theme_manager.colors["accent"]).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)

        # Сетка настроек (2 колонки)
        settings_grid = ttk.Frame(self.main_content)
        settings_grid.pack(fill="x", pady=0)
        
        # Левая колонка: Таймер и Системные настройки
        left_col = ttk.Frame(settings_grid)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Таймер автоматической остановки
        time_frame = ttk.LabelFrame(left_col, text=LANGUAGES[self.lang]["timer"], style="Card.TLabelframe", padding=15)
        time_frame.pack(fill="x", pady=(0, 15))
        
        t_inner = ttk.Frame(time_frame, style="Card.TFrame")
        t_inner.pack()
        ttk.Label(t_inner, text=LANGUAGES[self.lang]["hours"]).pack(side="left")
        self.spn_hours = ttk.Spinbox(t_inner, from_=0, to=23, width=3)
        self.spn_hours.set(0)
        self.spn_hours.pack(side="left", padx=5)
        ttk.Label(t_inner, text=LANGUAGES[self.lang]["minutes"]).pack(side="left", padx=(10, 0))
        self.spn_mins = ttk.Spinbox(t_inner, from_=0, to=59, width=3)
        self.spn_mins.set(0)
        self.spn_mins.pack(side="left", padx=5)

        # Системные параметры
        pref_frame = ttk.LabelFrame(left_col, text=LANGUAGES[self.lang]["system_settings"], style="Card.TLabelframe", padding=15)
        pref_frame.pack(fill="x")
        self.var_record = tk.BooleanVar(value=True)
        ttk.Checkbutton(pref_frame, text=LANGUAGES[self.lang]["record_video"], variable=self.var_record, style="Card.TCheckbutton").pack(anchor="w", pady=2)
        self.var_save_csv = tk.BooleanVar(value=True)
        ttk.Checkbutton(pref_frame, text=LANGUAGES[self.lang]["save_csv"], variable=self.var_save_csv, style="Card.TCheckbutton").pack(anchor="w", pady=2)
        self.var_debug = tk.BooleanVar(value=False)
        ttk.Checkbutton(pref_frame, text=LANGUAGES[self.lang]["debug_mode"], variable=self.var_debug, style="Card.TCheckbutton").pack(anchor="w", pady=2)
        
        # Параметр анти-дублирования (сек)
        f_anti = ttk.Frame(pref_frame, style="Card.TFrame")
        f_anti.pack(fill="x", pady=5)
        ttk.Label(f_anti, text=LANGUAGES[self.lang]["anti_duplication"]).pack(side="left")
        self.spn_anti = ttk.Spinbox(f_anti, from_=0.5, to=60, width=5, increment=0.5)
        self.spn_anti.set(2.0)
        self.spn_anti.pack(side="left", padx=5)

        ttk.Checkbutton(pref_frame, text=LANGUAGES[self.lang]["face_blur"], 
                        variable=self.var_face_blur, 
                        command=self.toggle_face_blur_pref, 
                        style="Card.TCheckbutton").pack(anchor="w", pady=2)

        # Переключатель GPU ускорения
        ttk.Checkbutton(pref_frame, text=LANGUAGES[self.lang]["gpu_acceleration"], 
                        variable=self.var_gpu_accel, 
                        command=self.toggle_gpu_accel, 
                        style="Card.TCheckbutton").pack(anchor="w", pady=2)

        # Переключатель Турбо-режима
        ttk.Checkbutton(pref_frame, text=LANGUAGES[self.lang]["turbo_mode"], 
                        variable=self.var_turbo_mode, 
                        command=self.toggle_turbo_mode, 
                        style="Card.TCheckbutton").pack(anchor="w", pady=2)

        # Правая колонка: Тепловая карта и Интеграции
        right_col = ttk.Frame(settings_grid)
        right_col.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Тепловая карта
        heatmap_frame = ttk.LabelFrame(right_col, text=LANGUAGES[self.lang]["heatmap"], style="Card.TLabelframe", padding=15)
        heatmap_frame.pack(fill="x", pady=(0, 15))
        self.var_save_heatmap = tk.BooleanVar(value=False)
        ttk.Checkbutton(heatmap_frame, text=LANGUAGES[self.lang]["save_heatmap_img"], variable=self.var_save_heatmap, style="Card.TCheckbutton").pack(anchor="w", pady=2)
        ttk.Button(heatmap_frame, text=LANGUAGES[self.lang]["heatmap_params"], command=self.open_heatmap_settings, style="Secondary.TButton").pack(fill="x", pady=5)

        # Интеграции (Google Drive и Оповещения)
        int_frame = ttk.LabelFrame(right_col, text="Integrations", style="Card.TLabelframe", padding=15)
        int_frame.pack(fill="x")

        # Google Drive статус и настройки
        gd_row = ttk.Frame(int_frame, style="Card.TFrame")
        gd_row.pack(fill="x", pady=5)
        ttk.Button(gd_row, text=LANGUAGES[self.lang]["google_drive"], command=self.open_gdrive_settings, style="Secondary.TButton").pack(side="left", fill="x", expand=True)
        self.gdrive_status_label = ttk.Label(gd_row, text="❌", width=3)
        self.gdrive_status_label.pack(side="right", padx=5)

        # Уведомления статус и настройки
        nt_row = ttk.Frame(int_frame, style="Card.TFrame")
        nt_row.pack(fill="x", pady=5)
        ttk.Button(nt_row, text=LANGUAGES[self.lang]["notifications"], command=self.open_notification_settings, style="Secondary.TButton").pack(side="left", fill="x", expand=True)
        self.notif_status_label = ttk.Label(nt_row, text="❌", width=3)
        self.notif_status_label.pack(side="right", padx=5)

        # === Кнопки управления зонами ===
        btn_frame = ttk.Frame(self.main_content, padding=(0, 20))
        btn_frame.pack(fill="x")
        
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["save_zones"], command=self.save_lines, style="Secondary.TButton").pack(side="left", expand=True, fill="x", padx=(0, 5))
        ttk.Button(btn_frame, text=LANGUAGES[self.lang]["load_zones"], command=self.load_lines, style="Secondary.TButton").pack(side="left", expand=True, fill="x", padx=(5, 0))

        # ГЛАВНАЯ КНОПКА: ЗАПУСК
        self.start_btn = ttk.Button(self.main_content, text=LANGUAGES[self.lang]["start_analysis"], command=self.start_engine)
        self.start_btn.pack(fill="x", pady=10, ipady=5)

        # Текст помощи (горячие клавиши)
        help_info = LANGUAGES[self.lang]["help_text"]
        ttk.Label(self.main_content, text=help_info, justify="left", font=("Segoe UI", 9), foreground=self.theme_manager.colors["gray"]).pack(pady=10)

    def set_language(self, lang_code):
        """
        Меняет язык интерфейса приложения.
        
        Args:
            lang_code (str): Код нового языка ('en', 'ru', 'kk').
        """
        if lang_code == self.lang:
            return
        self.lang = lang_code
        save_language(lang_code)
        messagebox.showinfo("Language", "Please restart the application for changes to take effect.")
        self.root.destroy()

    def _on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Update the width of the main_content to match the canvas"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def toggle_theme_ui(self):
        """Переключает тему оформления (Светлая/Темная) и обновляет значок на кнопке."""
        new_theme = self.theme_manager.toggle_theme()
        btn_text = "🌙" if new_theme == "light" else "☀️" # Луна для светлой темы (переход в темную), солнце для темной
        self.theme_btn.config(text=btn_text)
        self.theme_manager.setup_styles(self.root)
        self.theme_manager.apply_to_dialog(self.root)

    def apply_theme(self):
        self.theme_manager.setup_styles(self.root)
        self.theme_manager.apply_to_dialog(self.root) # updates non-ttk widgets if any

        # Update specific non-ttk widget colors
        # e.g. if we had canvases or text widgets in main window
        pass

    def toggle_theme(self):
        self.theme_manager.toggle_theme()
        self.apply_theme()
        # Refresh current window widgets to ensure style redraw
        # Note: TTK styles usually update automatically, but standard TK widgets need manual config
        # We might need to restart or aggressively reconfigure for full effect on complex layouts
        # Ideally, we just call apply_theme which re-configures the style object
        
        # Update specific labels like gdrive/notif status that act like badges
        self.update_gdrive_status()
        self.update_notification_status()

    def update_gdrive_status(self):
        """Обновляет индикатор (иконку/текст) статуса Google Drive в главном меню."""
        try:
            if self.google_drive.enabled and self.google_drive.folder_id:
                if hasattr(self.google_drive, 'service') and self.google_drive.service:
                    text = LANGUAGES[self.lang]["gdrive_active"]
                    color = self.theme_manager.colors["success"]
                else:
                    text = LANGUAGES[self.lang]["gdrive_auth_needed"]
                    color = self.theme_manager.colors["warning"]
            else:
                text = LANGUAGES[self.lang]["gdrive_not_configured"]
                color = self.theme_manager.colors["error"]
            
            self.gdrive_status_label.config(text=text, foreground=color)
        except Exception as e:
            print(f"Error updating GDrive status: {e}")
            self.gdrive_status_label.config(text="❌", foreground=self.theme_manager.colors["error"])


    def open_gdrive_settings(self):
        """Открывает диалог настроек Google Drive."""
        dialog = GoogleDriveSettingsDialog(self.root, self.google_drive, self.lang)
        self.theme_manager.apply_to_dialog(dialog.dialog) # Apply theme to dialog
        dialog.show()
        self.update_gdrive_status()

    def update_notification_status(self):
        """Обновляет индикатор статуса уведомлений в главном меню."""
        try:
            email_status = LANGUAGES[self.lang]["email_on"] if self.notification_system.email_enabled else LANGUAGES[self.lang]["email_off"]
            color = self.theme_manager.colors["success"] if self.notification_system.email_enabled else self.theme_manager.colors["error"]
            self.notif_status_label.config(
                text=f"{email_status}",
                foreground=color
            )
        except Exception as e:
            print(f"Error updating notification status: {e}")
            self.notif_status_label.config(text="❌", foreground=self.theme_manager.colors["error"])


    def open_notification_settings(self):
        """Открывает диалог настроек уведомлений."""
        dialog = NotificationSettingsDialog(self.root, self.notification_system, self.lang)
        self.theme_manager.apply_to_dialog(dialog.dialog)
        # Update text widgets in dialog manually as they are Tk
        dialog.email_contacts.configure(bg=self.theme_manager.colors["input_bg"], fg=self.theme_manager.colors["input_fg"])
        
        dialog.show()
        self.update_notification_status()

    def open_heatmap_settings(self):
        dialog = HeatmapSettingsDialog(self.root, self, self.lang)
        self.theme_manager.apply_to_dialog(dialog.dialog)
        dialog.show()

    def toggle_face_blur_pref(self):
        """Переключает состояние размытия лиц из UI."""
        self.face_blur_enabled = self.var_face_blur.get()
        print(f"Face Blur: {'ENABLED' if self.face_blur_enabled else 'DISABLED'}")

    def init_torch_device(self):
        """Инициализирует устройство для PyTorch (CPU или GPU) в зависимости от настроек и доступности."""
        user_wants_gpu = self.var_gpu_accel.get()
        
        if user_wants_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            self.use_fp16 = True # Включаем половинную точность для GPU
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"AI People Counter using GPU: {gpu_name} (FP16: ON)")
            except:
                print(f"AI People Counter using GPU (Device: cuda, FP16: ON)")
        else:
            self.device = 'cpu'
            self.use_fp16 = False
            if not user_wants_gpu:
                print("AI People Counter using CPU (GPU acceleration disabled by user).")
            else:
                print("AI People Counter fallback to CPU (CUDA not available).")
                if torch.cuda.device_count() > 0:
                    print("Note: CUDA devices found but torch.cuda.is_available() is False. Check drivers/PyTorch version.")

    def toggle_gpu_accel(self):
        """Обрабатывает переключение GPU ускорения через UI."""
        enabled = self.var_gpu_accel.get()
        
        # Сохранение настройки
        config = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except:
                pass
        config["gpu_accel"] = enabled
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except:
            pass
            
        # Обновление устройства
        self.init_torch_device()
        
        # Перенос модели на новое устройство
        try:
            self.model.to(self.device)
            print(f"Model moved to {self.device}")
        except Exception as e:
            print(f"Error moving model to {self.device}: {e}")

    def toggle_turbo_mode(self):
        """Переключает Турбо-режим (смена модели и разрешения)."""
        is_turbo = self.var_turbo_mode.get()
        model_name = "yolov8m.pt" if is_turbo else "yolov8n.pt"
        
        print(f"Switching to Turbo Mode: {'ON' if is_turbo else 'OFF'} (Model: {model_name})")
        
        # Пересоздаем модель с новыми параметрами
        try:
            # Освобождаем память старой модели
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = YOLO(model_name).to(self.device)
            print(f"Model {model_name} loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error switching model for Turbo Mode: {e}")
            # Откат на Nano при ошибке
            self.model = YOLO("yolov8n.pt").to(self.device)
            self.var_turbo_mode.set(False)
            messagebox.showwarning("GPU", f"Error switching device: {e}")


    def toggle_source(self):
        """
        Переключает доступность элементов управления в зависимости от выбранного источника видео.
        """
        mode = self.source_type.get()
        if mode == "webcam":
            self.spn_cam_idx.config(state="normal")
            self.ent_ip_url.config(state="disabled")
            self.btn_select_file.config(state="disabled")
        elif mode == "ip":
            self.spn_cam_idx.config(state="disabled")
            self.ent_ip_url.config(state="normal")
            self.btn_select_file.config(state="disabled")
        elif mode == "file":
            self.spn_cam_idx.config(state="disabled")
            self.ent_ip_url.config(state="disabled")
            self.btn_select_file.config(state="normal")

    def select_video_file(self):
        """Открывает диалог выбора видеофайла."""
        file_path = filedialog.askopenfilename(
            title=LANGUAGES[self.lang]["select_file"],
            filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")]
        )
        if file_path:
            self.selected_file_path.set(os.path.basename(file_path))
            self.video_file_path = file_path

    def save_lines(self):
        """
        Save current zones and lines to a JSON file.
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Сохранить зоны",
            initialfile="config_v21.json"
        )
        if not file_path:
            return
        try:
            data = []
            for line in self.lines:
                if line.type == "zone" and line.points:
                    data.append({
                        "type": "zone",
                        "points": line.points
                    })
                else:
                    data.append({
                        "p1": line.p1,
                        "p2": line.p2,
                        "type": line.type
                    })
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo(LANGUAGES[self.lang]["success"], LANGUAGES[self.lang]["zones_saved"].format(file_path))
        except Exception as e:
            messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["file_load_error"].format(str(e)))

    def load_lines(self):
        """
        Load zones and lines from a JSON file.
        """
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Загрузить зоны"
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.lines = []
            for item in data:  # ИСПРАВЛЕНО: добавлено "data"
                if item.get("type") == "zone" and "points" in item:
                    zone = CounterLine(None, None, "zone", item["points"])
                    zone.calculate_bounds()
                    self.lines.append(zone)
                elif "p1" in item and "p2" in item and "type" in item:
                    self.lines.append(CounterLine(
                        tuple(item["p1"]),
                        tuple(item["p2"]),
                        item["type"]
                    ))
            messagebox.showinfo(LANGUAGES[self.lang]["success"], f"Загружено {len(self.lines)} зон/линий из файла:\n{file_path}")
        except json.JSONDecodeError:
            messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["json_corrupted"])
        except Exception as e:
            messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["file_load_error"].format(str(e)))

    def start_engine(self):
        """
        Запускает процесс обработки видео в отдельном потоке.
        Выбирает метод (живое видео или файл) в зависимости от настроек.
        """
        if self.video_processing:
            return
        
        mode = self.source_type.get()
        if mode == "file" and not hasattr(self, 'video_file_path'):
            messagebox.showwarning(LANGUAGES[self.lang]["warning"], LANGUAGES[self.lang]["no_video_selected"])
            return
        if self.notification_system.email_enabled and not self.notification_system.contacts['emails']:
            if not messagebox.askyesno("⚠️ Внимание", LANGUAGES[self.lang]["no_contacts_warning"]):
                return

        self.video_processing = True
        self.abort_processing = False
        self.start_btn.config(state="disabled") # Assuming start_btn is the button to start
        # self.btn_stop.config(state="normal") # Assuming there's a stop button
        
        # Сброс счетчиков перед началом
        for key in self.counts:
            self.counts[key] = 0
        
        # Reset individual line/zone counters
        for line in self.lines:
            line.count = 0
            
        # Обновляем состояния тепловой карты перед запуском
        self.heatmap_enabled = False  # Тепловая карта теперь отключена по умолчанию
        self.save_heatmap = self.var_save_heatmap.get()
        self.save_csv = self.var_save_csv.get()
        
        # Запуск в отдельном потоке, чтобы GUI не завис
        thread = threading.Thread(target=self._start_engine_thread, daemon=True)
        thread.start()

    def stop_engine(self):
        """Сигнализирует потоку обработки видео о необходимости остановиться."""
        self.abort_processing = True
        self.video_processing = False
        self.start_btn.config(state="normal") # Assuming start_btn is the button to start
        # self.btn_stop.config(state="disabled") # Assuming there's a stop button

    def toggle_heatmap_mode(self):
        """Включает/выключает отображение тепловой карты в реальном времени."""
        self.heatmap_enabled = not self.heatmap_enabled
        status = "ON" if self.heatmap_enabled else "OFF"
        print(f"Heatmap: {status}")

    def _start_engine_thread(self):
        """Internal method to run the processing loop in a separate thread."""
        try:
            if self.source_type.get() == "file":
                self.process_video_file()
            else:
                self.process_live_video()
        finally:
            self.video_processing = False
            self.root.after(0, lambda: self.start_btn.config(state="normal"))

    def mouse_callback_live(self, event, x, y, flags, param):
        """
        Handle mouse events for drawing lines and zones in the live video window.
        
        Args:
            event: The mouse event type.
            x (int): X-coordinate of the mouse event.
            y (int): Y-coordinate of the mouse event.
            flags: Event flags.
            param: User-defined parameter.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_draw_mode == "zone":
                if not self.drawing_line:
                    self.drawing_line.append((x, y))
                else:
                    first_point = self.drawing_line[0]
                    if math.hypot(x - first_point[0], y - first_point[1]) < 15:
                        if len(self.drawing_line) >= 3:
                            zone_id = len([l for l in self.lines if l.type == "zone"]) + 1
                            zone = CounterLine(None, None, "zone", self.drawing_line.copy(), zone_id=zone_id)
                            zone.calculate_bounds()
                            self.lines.append(zone)
                            self.drawing_line = []
                        else:
                            self.drawing_line.append((x, y))
                    else:
                        self.drawing_line.append((x, y))
            else:
                self.drawing_line.append((x, y))
                if len(self.drawing_line) == 2:
                    self.lines.append(CounterLine(self.drawing_line[0], self.drawing_line[1], self.current_draw_mode))
                    self.drawing_line = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.lines:
                self.lines.pop()
            elif self.drawing_line:
                self.drawing_line = []

    # --- Новые методы для тепловой карты ---
    def toggle_heatmap(self):
        """
        Toggle the heatmap visualization on or off.
        """
        self.heatmap_enabled = not self.heatmap_enabled
        print(f"Тепловая карта {'включена' if self.heatmap_enabled else 'выключена'}")

    def update_heatmap(self, frame_shape, detections):
        """
        Update the heatmap based on current detections.
        
        Args:
            frame_shape (tuple): Shape of the current frame (height, width).
            detections (list): List of bounding boxes for detected people.
        """
        if not self.heatmap_enabled and not self.save_heatmap:
            return
        h, w = frame_shape[:2]
        with self.heatmap_lock:
            if self.heatmap is None or self.heatmap.shape[:2] != (h, w):
                self.heatmap = np.zeros((h, w), dtype=np.float32)
            for det in detections:
                x1, y1, x2, y2 = map(int, det)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # Увеличиваем интенсивность в центральной точке
                if 0 <= cy < h and 0 <= cx < w:
                    self.heatmap[cy, cx] += 1.0

    def apply_heatmap(self, frame):
        """
        Overlay the heatmap on the given frame.
        
        Applies Gaussian blur and colormap to the accumulated heatmap data
        and blends it with the original frame.
        
        Args:
            frame (numpy.ndarray): The current video frame.
            
        Returns:
            numpy.ndarray: The frame with the heatmap overlay (if enabled).
        """
        if not self.heatmap_enabled or self.heatmap is None:
            return frame
        h, w = frame.shape[:2]
        with self.heatmap_lock:
            # Применяем размытие для "распространения" тепла
            blurred_heatmap = cv2.GaussianBlur(self.heatmap, self.heatmap_blur_kernel, 0)
            # Нормализуем для отображения
            norm_heatmap = np.uint8(255 * blurred_heatmap / (blurred_heatmap.max() + 1e-6))
            # Создаем цветную тепловую карту
            colored_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
            # Накладываем на фрейм
            overlay = cv2.addWeighted(frame, 1 - self.heatmap_alpha, colored_heatmap, self.heatmap_alpha, 0)
            # Обновляем основную тепловую карту с учетом затухания
            self.heatmap *= self.heatmap_decay_factor
            return overlay

    def save_heatmap_image(self, filename_suffix="heatmap"):
        if self.heatmap is None:
            print("Тепловая карта не содержит данных.")
            return
        with self.heatmap_lock:
            if self.heatmap.max() == 0:
                print("Тепловая карта пуста.")
                return
            # Нормализуем для сохранения
            norm_heatmap = np.uint8(255 * self.heatmap / self.heatmap.max())
            colored_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(BASE_DIR, f"heatmap_{filename_suffix}_{timestamp}.png")
            cv2.imwrite(filename, colored_heatmap)
            print(f"Изображение тепловой карты сохранено: {filename}")
            return filename

    def process_live_video(self):
        """
        Main loop for processing video feeds (Webcam, IP Camera, or File).
        
        Handles:
        - Frame capture and resizing.
        - Object tracking and counting.
        - GUI updates (Heatmap, HUD).
        - User input during processing.
        """
        mode = self.source_type.get()
        if mode == "file":
            source = self.video_file_path
            # First, get the initial frame for markup
            cap_init = cv2.VideoCapture(source)
            ret_init, first_frame = cap_init.read()
            cap_init.release()

            if not ret_init:
                messagebox.showerror(LANGUAGES[self.lang]["error"], "Can't read video file")
                self.stop_engine()
                return
            
            # Open markup window
            self.markup_confirmed = False
            self.create_markup_window(first_frame)
            self.root.wait_window(self.markup_window)
            
            if not self.markup_confirmed:
                self.stop_engine()
                return
            
            # Hide main window during file processing
            self.root.after(0, lambda: self.root.withdraw())
            window_name = "PRO People Counter V2.1 - FILE PROCESSING"
            cv2.namedWindow(window_name)
            # No mouse callback for file processing as markup is done upfront
            
        else: # webcam or ip_camera
            if mode == "webcam":
                source = int(self.spn_cam_idx.get())
            else:
                source = self.ent_ip_url.get()
            
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["video_source_error"]))
                return
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                return
            
            window_name = "PRO People Counter V2.1 - LIVE"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.mouse_callback_live)
            
            # Maximize window and bring to front
            try:
                hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 3) # SW_MAXIMIZE = 3
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception as e:
                print(f"Window focus error: {e}")

        total_seconds = int(self.spn_hours.get()) * 3600 + int(self.spn_mins.get()) * 60
        anti_double = float(self.spn_anti.get())
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["video_source_error"]))
            self.stop_engine()
            return

        ret, frame_for_dims = cap.read()
        if not ret:
            cap.release()
            self.stop_engine()
            return

        h, w = frame_for_dims.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1 or fps > 100: fps = 30.0
        
        start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.join(BASE_DIR, f"video_{start_ts}.mp4")
        csv_name = os.path.join(BASE_DIR, f"report_{start_ts}.csv")
        # Списки для отслеживания созданных файлов
        created_files = []
        if self.save_csv:
            with open(csv_name, 'w', newline='', encoding='utf-8') as f:
                header = ['Time', 'Entered', 'Exited', 'Control', 'Zone Total', 'Total People']
                # Add headers for fixed number of zones (e.g., 10)
                for i in range(1, 11):
                    header.append(f"Zone {i}")
                csv.writer(f).writerow(header)
            created_files.append(csv_name)
        writer = None
        if self.var_record.get():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_name, fourcc, fps, (w, h))
            created_files.append(video_name)
        self.root.after(0, lambda: self.root.withdraw())
        cv2.namedWindow("PRO People Counter V2.1 - LIVE")
        cv2.setMouseCallback("PRO People Counter V2.1 - LIVE", self.mouse_callback_live)
        
        # Maximize window and bring to front
        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, "PRO People Counter V2.1 - LIVE")
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 3) # SW_MAXIMIZE = 3
                ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception as e:
            print(f"Window focus error: {e}")

        start_time = time.time()
        self.abort_processing = False
        # Инициализация last_counts для отслеживания изменений
        last_counts = self.counts.copy()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or self.abort_processing:
                    break
                elapsed = time.time() - start_time
                if total_seconds > 0 and elapsed > total_seconds:
                    break
                
                # Подсчет FPS
                current_time = time.time()
                dt = current_time - self.last_frame_time
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                self.last_frame_time = current_time
                # === БЛОК ДЕТЕКЦИИ И ТРЕКИНГА (YOLOv8) ===
                # Определяем параметры инференса в зависимости от режима (Турбо или Нормальный)
                is_turbo = self.var_turbo_mode.get()
                imgsz = 1280 if is_turbo else 640 # В Турбо увеличиваем разрешение до 1280px для мелких деталей
                conf = 0.4 if is_turbo else 0.3 # Порог уверенности: 0.4 оптимален для баланса точности и шума
                # IoU (Intersection over Union): порог перекрытия рамок.
                # 0.4 в Турбо позволяет эффективнее "склеивать" перекрывающиеся детекции одного человека на крупных планах.
                iou = 0.4 if is_turbo else 0.7 
                
                # Запуск трекера Ultralytics
                results = self.model.track(
                    frame, 
                    persist=True,        # Сохранять ID объектов между кадрами
                    classes=[0],         # Только класс "человек" (person)
                    verbose=False,       # Отключить логгирование в консоль для скорости
                    conf=conf, 
                    iou=iou,
                    device=self.device,  # CUDA или CPU (автоматический выбор)
                    half=self.use_fp16, # Использование FP16 (половинная точность) для GPU (ускоряет в 2-3 раза)
                    imgsz=imgsz
                )
                total_people = 0
                detections = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    total_people = len(ids)
                    detections = boxes 
                    for box, obj_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        # Размытие лиц (анонимизация)
                        if self.face_blur_enabled:
                            person_h, person_w = y2 - y1, x2 - x1
                            h_frame, w_frame = frame.shape[:2]
                            y1_h, y2_h = max(0, y1), min(h_frame, y1 + int(person_h * 0.8))
                            x1_h, x2_h = max(0, x1), min(w_frame, x2)
                            head_roi = frame[y1_h:y2_h, x1_h:x2_h]
                            if head_roi.size > 0:
                                gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                                min_f = 60 if is_turbo else 30
                                faces = self.face_cascade.detectMultiScale(gray_head, 1.05, 5, minSize=(min_f, min_f))
                                if len(faces) > 0:
                                    for (fx, fy, fw, fh) in faces:
                                        fx1, fy1 = x1_h + fx, y1_h + fy
                                        fx2, fy2 = fx1 + fw, fy1 + fh
                                        pad = int(fw * 0.1)
                                        fx1_p, fy1_p = max(0, fx1 - pad), max(0, fy1 - pad)
                                        fx2_p, fy2_p = min(w_frame, fx2 + pad), min(h_frame, fy2 + pad)
                                        face_area = frame[fy1_p:fy2_p, fx1_p:fx2_p]
                                        k_size = int(fw * 0.8) | 1
                                        if k_size < 15: k_size = 15
                                        frame[fy1_p:fy2_p, fx1_p:fx2_p] = cv2.GaussianBlur(face_area, (k_size, k_size), 0)
                                else:
                                    f_w = int(person_w * 0.6)
                                    f_h = int(f_w * 1.3)
                                    x_c = (x1 + x2) // 2
                                    y_o = int(person_h * 0.18)
                                    y1_b, y2_b = max(0, y1 + y_o), min(h_frame, y1 + y_o + f_h)
                                    x1_b, x2_b = max(0, x_c - f_w // 2), min(w_frame, x_c + f_w // 2)
                                    if y2_b > y1_b and x2_b > x1_b:
                                        k_size = int(f_w * 0.8) | 1
                                        if k_size < 15: k_size = 15
                                        frame[y1_b:y2_b, x1_b:x2_b] = cv2.GaussianBlur(frame[y1_b:y2_b, x1_b:x2_b], (k_size, k_size), 0)
                        
                        # === БЛОК ТРЕКИНГА И ЗОН ===
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        # Получаем предыдущее состояние объекта для анализа пересечения линий
                        old_state = self.tracked_objects.get(obj_id, (cx, cy, None))
                        if len(old_state) == 3 and isinstance(old_state[2], bool):
                             # Миграция состояния из версий 2.x (если была булевая метка входа)
                             old_cx, old_cy, old_zone_id = old_state[0], old_state[1], None
                        else:
                             old_cx, old_cy, old_zone_id = old_state

                        current_zone_id = None 
                        in_any_zone = False
                        
                        # Проверка нахождения в зонах интереса (Polygon intersection)
                        for line in self.lines:
                            if line.type == "zone":
                                if line.point_in_polygon((cx, cy)):
                                    in_any_zone = True
                                    current_zone_id = line.zone_id
                                    # Срабатывает при ПЕРВОМ входе или смене зоны
                                    if old_zone_id != line.zone_id:
                                        now = time.time()
                                        # Анти-дублирование (таймаут) предотвращает ложные срабатывания при дрожании детекции
                                        if obj_id not in self.last_seen_time or (now - self.last_seen_time[obj_id]) > anti_double:
                                            self.counts["zone"] += 1     
                                            line.count += 1              
                                            self.last_seen_time[obj_id] = now
                                    break 
                        
                        if not in_any_zone:
                            current_zone_id = None

                        # Проверка пересечения линий (ВХОД / ВЫХОД / КОНТРОЛЬ)
                        for line in self.lines:
                            if line.type != "zone" and line.is_intersecting((old_cx, old_cy), (cx, cy)):
                                now = time.time()
                                if obj_id not in self.last_seen_time or (now - self.last_seen_time[obj_id]) > anti_double:
                                    if line.type == "entry":
                                        self.counts["entered"] += 1
                                    elif line.type == "exit":
                                        self.counts["exited"] += 1
                                    else:
                                        self.counts["control"] += 1
                                    self.last_seen_time[obj_id] = now
                                    break
                        # Обновляем состояние объекта в памяти
                        self.tracked_objects[obj_id] = (cx, cy, current_zone_id)
                else:
                    # === PROXIMITY FIX (50 см) ===
                    # Если YOLO не нашел людей (например, только голова в кадре), 
                    # ищем лица по всему кадру для гарантии приватности (размытия).
                    if self.face_blur_enabled:
                         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                         min_f = 100 if is_turbo else 50 # Вблизи лица крупные
                         faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_f, min_f))
                         for (fx, fy, fw, fh) in faces:
                             pad = int(fw * 0.1) # Паддинг 10% для надежного скрытия ушей/волос
                             fx1, fy1 = max(0, fx - pad), max(0, fy - pad)
                             fx2, fy2 = min(frame.shape[1], fx + fw + pad), min(frame.shape[0], fy + fh + pad)
                             face_area = frame[fy1:fy2, fx1:fx2]
                             if face_area.size > 0:
                                 # Динамическое ядро размытия: зависит от физического размера лица на матрице
                                 k_size = int(fw * 0.8) | 1
                                 if k_size < 15: k_size = 15
                                 frame[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(face_area, (k_size, k_size), 0)
                self.counts["total"] = total_people
                # Update heatmap before any drawing
                if self.heatmap_enabled or self.save_heatmap:
                    self.update_heatmap(frame.shape, detections)
                frame = self.apply_heatmap(frame)
                # Check notification thresholds
                self.notification_system.check_thresholds_and_notify(self.counts)
                
                # OPTIMIZED CSV LOGGING: Write only if counts changed
                if self.save_csv:
                    # Check global counts change
                    global_changed = (self.counts["entered"] != last_counts["entered"] or
                                   self.counts["exited"] != last_counts["exited"] or
                                   self.counts["control"] != last_counts["control"] or
                                   self.counts["zone"] != last_counts["zone"] or
                                   self.counts["total"] != last_counts["total"])
                    
                    # Check individual zone counts change
                    # We need to track last known zone counts to detect changes there too
                    # For simplicity, we can rely on global zone count change OR check current line counts
                    # But since we want to log ALL zone counts if ANY changed, we iterate.
                    
                    # Let's verify if any specific zone count changed.
                    # To do this efficiently without storing a separate dict of last zone counts every frame,
                    # we can iterate lines. But we also need to know if we *should* write.
                    
                    # We will assume if global 'zone' count changed, then some specific zone changed.
                    # This is true because global is sum of events (or just incremented on any zone entry).
                    
                    if global_changed: 
                        row_data = [
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            self.counts["entered"],
                            self.counts["exited"],
                            self.counts["control"],
                            self.counts["zone"],
                            self.counts["total"]
                        ]
                        
                        # Add individual zone counts to row, matching the pre-allocated headers (Zone 1 to 10)
                        current_zones = [line for line in self.lines if line.type == "zone"]
                        for i in range(1, 11): # Zones 1 to 10
                            zone_count = 0
                            # Find zone with this ID or index
                            for z in current_zones:
                                z_id = z.zone_id if z.zone_id else (self.lines.index(z) + 1) # Fallback to index if no ID
                                if z_id == i:
                                    zone_count = z.count
                                    break
                            row_data.append(zone_count)

                        with open(csv_name, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow(row_data)
                        
                        last_counts = self.counts.copy()
                if self.var_debug.get():
                    if results[0].boxes is not None:
                        for box in results[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 1)
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                # ПАКЕТНАЯ ОТРИСОВКА ЗОН (за один проход)
                overlay = frame.copy()
                for line in self.lines:
                    if line.type == "zone" and line.points:
                        pts = np.array(line.points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], True, line.color, 2)
                        cv2.fillPoly(overlay, [pts], line.color)
                
                # Смешиваем все зоны разом
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, dst=frame)
                
                # ПАКЕТНАЯ ОТРИСОВКА ТЕКСТА
                batch_labels = []
                for line in self.lines:
                    if line.type == "zone" and line.points:
                        cx = sum(p[0] for p in line.points) // len(line.points)
                        cy = sum(p[1] for p in line.points) // len(line.points)
                        batch_labels.append({'text': LANGUAGES[self.lang]["interest_zone"], 'x': cx - 50, 'y': cy - 10, 'color': line.color})
                    else:
                        cv2.line(frame, line.p1, line.p2, line.color, 3)
                        batch_labels.append({'text': line.type.upper(), 'x': line.p1[0], 'y': line.p1[1]-5, 'color': line.color})
                if self.current_draw_mode == "zone" and self.drawing_line:
                    if len(self.drawing_line) >= 2:
                        pts = np.array(self.drawing_line, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, (255, 0, 255), 2)
                        for point in self.drawing_line:
                            cv2.circle(frame, point, 5, (255, 0, 255), -1)
                    else:
                        if len(self.drawing_line) == 1:
                            cv2.circle(frame, self.drawing_line[0], 5, (0, 255, 0), -1)
                        elif len(self.drawing_line) == 2:
                            cv2.line(frame, self.drawing_line[0], self.drawing_line[1], (0, 255, 255), 2)
                # ПАКЕТНАЯ ОТРИСОВКА ВСЕГО ТЕКСТА (HUD + Зоны)
                batch_labels += self.get_hud_labels(frame, total_seconds - elapsed if total_seconds > 0 else None)
                frame = self.draw_text_batch(frame, batch_labels)
                if writer:
                    writer.write(frame)
                cv2.imshow("PRO People Counter V2.1 - LIVE", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.abort_processing = True
                    break
                elif key == ord('1'):
                    self.current_draw_mode = "entry"
                elif key == ord('2'):
                    self.current_draw_mode = "exit"
                elif key == ord('3'):
                    self.current_draw_mode = "control"
                elif key == ord('4'):
                    self.current_draw_mode = "zone"
                    self.drawing_line = []
                elif key == ord('5'): # Toggle heatmap
                    self.toggle_heatmap()
                elif key == ord('s'):
                    self.save_lines()
                elif key == ord('l'):
                    self.load_lines()
                elif key == ord('c'):
                    self.drawing_line = []
                elif key == ord('n'):
                    self.notification_system.check_thresholds_and_notify(self.counts, force=True)
                elif key == ord('g'):
                    self.root.after(0, self.open_gdrive_settings)
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.root.after(0, lambda: self.root.deiconify())
        # Save heatmap image at the end if enabled
        heatmap_file = None
        if self.save_heatmap:
            heatmap_file = self.save_heatmap_image("live_session")
            if heatmap_file:
                created_files.append(heatmap_file)
        # Автоматическая загрузка файлов в Google Drive
        if self.google_drive.enabled and self.google_drive.folder_id:
            self.upload_all_files_to_google_drive(created_files, "live_session")
        message = f"{LANGUAGES[self.lang]['files_saved_title']}\n"
        message += LANGUAGES[self.lang]["video_timelapse"].format(video_name)
        if self.save_csv:
            message += f"\n{LANGUAGES[self.lang]['csv_report'].format(csv_name)}"
        else:
            message += f"\n{LANGUAGES[self.lang]['csv_disabled']}"
        if self.google_drive.enabled and self.google_drive.folder_id:
            message += LANGUAGES[self.lang]["gdrive_uploaded"]
        if self.notification_system.email_enabled:
            message += LANGUAGES[self.lang]["notifications_sent"].format("") # Just notification message
        self.root.after(0, lambda: messagebox.showinfo(LANGUAGES[self.lang]["success"], message))

    def draw_text_batch(self, frame, labels):
        """
        Draw multiple text labels on the frame in a single PIL session for performance.
        
        Args:
            frame (numpy.ndarray): The video frame.
            labels (list): List of dicts/tuples with {'text', 'x', 'y', 'color'}.
        """
        if not labels:
            return frame
            
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        outline_thickness = 1
        for label in labels:
            text = label['text']
            x, y = label['x'], label['y']
            color = label['color'][::-1] # BGR to RGB
            
            # Outline
            for dx in range(-outline_thickness, outline_thickness + 1):
                for dy in range(-outline_thickness, outline_thickness + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
            # Text
            draw.text((x, y), text, font=font, fill=color)
            
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_text_with_cyrillic(self, frame, text, x, y, color):
        """Wrapper for backward compatibility, usage is deprecated for batching."""
        return self.draw_text_batch(frame, [{'text': text, 'x': x, 'y': y, 'color': color}])

    def process_video_file(self):
        """
        Main loop for processing video files.
        
        Features:
        - Optimization options (resize/fps reduction).
        - Progress tracking with a progress bar.
        - Accelerated processing (faster than real-time if possible).
        """
        optimize = messagebox.askyesno(
            LANGUAGES[self.lang]["optimization_title"],
            LANGUAGES[self.lang]["optimization_msg"]
        )
        cap = cv2.VideoCapture(self.video_file_path)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["video_open_error"]))
            return
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_width, target_height = orig_width, orig_height
        target_fps = orig_fps
        frame_step = 1
        if optimize:
            target_width, target_height = 640, 480
            target_fps = 30.0
            frame_step = max(1, int(round(orig_fps / target_fps)))
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            self.root.after(0, lambda: messagebox.showerror(LANGUAGES[self.lang]["error"], LANGUAGES[self.lang]["first_frame_error"]))
            return
        if optimize:
            first_frame = cv2.resize(first_frame, (target_width, target_height))
        self.create_markup_window(first_frame.copy())
        self.root.wait_window(self.markup_window)
        if not self.markup_confirmed:
            cap.release()
            return
        frames_to_process = math.ceil(total_frames / frame_step)
        estimated_fps = 8 if optimize else 2
        estimated_seconds = frames_to_process / estimated_fps
        hours = int(estimated_seconds // 3600)
        minutes = int((estimated_seconds % 3600) // 60)
        seconds = int(estimated_seconds % 60)
        time_str = ""
        if hours > 0:
            time_str += f"{hours} {LANGUAGES[self.lang]['hours_unit']} "
        if minutes > 0:
            time_str += f"{minutes} {LANGUAGES[self.lang]['mins_unit']} "
        time_str += f"{seconds} {LANGUAGES[self.lang]['secs_unit']}"
        proceed = messagebox.askyesno(
            LANGUAGES[self.lang]["time_est_title"],
            f"{LANGUAGES[self.lang]['total_frames'].format(total_frames)}\n"
            f"{LANGUAGES[self.lang]['frames_to_process'].format(frames_to_process)}\n"
            f"{LANGUAGES[self.lang]['est_time'].format(time_str)}\n"
            f"{LANGUAGES[self.lang]['continue_q']}"
        )
        if not proceed:
            cap.release()
            return
        progress_window = tk.Toplevel(self.root)
        progress_window.title(LANGUAGES[self.lang]["processing_video"].rstrip('.'))
        self.theme_manager.apply_to_dialog(progress_window)
        progress_window.transient(self.root)
        progress_window.grab_set()
        ttk.Label(progress_window, text=LANGUAGES[self.lang]["processing_video"], font=("Arial", 10, "bold")).pack(pady=10)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill="x", padx=20, pady=5)
        progress_label = ttk.Label(progress_window, text="0%")
        progress_label.pack()
        cancel_btn = ttk.Button(progress_window, text=LANGUAGES[self.lang]["cancel"], command=lambda: setattr(self, 'abort_processing', True))
        cancel_btn.pack(pady=10)
        center_window(progress_window)
        start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.join(BASE_DIR, f"timelapse_{start_ts}.mp4")
        csv_name = os.path.join(BASE_DIR, f"report_{start_ts}.csv")
        # Списки для отслеживания созданных файлов
        created_files = []
        if self.save_csv:
            with open(csv_name, 'w', newline='', encoding='utf-8') as f:
                header = ['Time', 'Entered', 'Exited', 'Control', 'Zone Total', 'Total People']
                # Add headers for fixed number of zones (e.g., 10)
                for i in range(1, 11):
                    header.append(f"Zone {i}")
                csv.writer(f).writerow(header)
            created_files.append(csv_name)
        writer = None
        if self.var_record.get():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_name, fourcc, target_fps, (target_width, target_height))
            created_files.append(video_name)
        # Reset counts and objects for new video
        self.counts = {"entered": 0, "exited": 0, "control": 0, "zone": 0, "total": 0}
        self.tracked_objects = {}
        self.last_seen_time = {}
        anti_double = float(self.spn_anti.get())
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        processed_frames = 0
        start_time = time.time()
        self.abort_processing = False
        last_notification_check = time.time()
        notification_check_interval = 10
        # Инициализация last_counts для отслеживания изменений
        last_counts = self.counts.copy()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or self.abort_processing:
                    break
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_id % frame_step != 0:
                    continue
                if optimize:
                    frame = cv2.resize(frame, (target_width, target_height))
                # Инференс (с поддержкой Турбо-режима и NMS)
                is_turbo = self.var_turbo_mode.get()
                imgsz = 1280 if is_turbo else 640
                conf = 0.4 if is_turbo else 0.3
                iou = 0.4 if is_turbo else 0.7
                results = self.model.track(frame, persist=True, classes=[0], verbose=False, conf=conf, iou=iou, device=self.device, half=self.use_fp16, imgsz=imgsz)
                
                total_people = 0
                detections = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    total_people = len(ids)
                    detections = boxes
                    for box, obj_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        if self.face_blur_enabled:
                            person_h, person_w = y2 - y1, x2 - x1
                            h_f, w_f = frame.shape[:2]
                            y1_h, y2_h = max(0, y1), min(h_f, y1 + int(person_h * 0.8))
                            x1_h, x2_h = max(0, x1), min(w_f, x2)
                            head_roi = frame[y1_h:y2_h, x1_h:x2_h]
                            if head_roi.size > 0:
                                gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                                min_f = 60 if is_turbo else 30
                                faces = self.face_cascade.detectMultiScale(gray_head, 1.05, 5, minSize=(min_f, min_f))
                                if len(faces) > 0:
                                    for (fx, fy, fw, fh) in faces:
                                        fx1, fy1 = x1_h + fx, y1_h + fy
                                        fx2, fy2 = fx1 + fw, fy1 + fh
                                        pad = int(fw * 0.1)
                                        f1x, f1y = max(0, fx1 - pad), max(0, fy1 - pad)
                                        f2x, f2y = min(w_f, fx2 + pad), min(h_f, fy2 + pad)
                                        face_area = frame[f1y:f2y, f1x:f2x]
                                        if face_area.size > 0:
                                            k_size = int(fw * 0.8) | 1
                                            if k_size < 15: k_size = 15
                                            frame[f1y:f2y, f1x:f2y] = cv2.GaussianBlur(face_area, (k_size, k_size), 0)
                                else:
                                    f_w = int(person_w * 0.6)
                                    f_h = int(f_w * 1.3)
                                    x_c = (x1 + x2) // 2
                                    y_o = int(person_h * 0.18)
                                    y1_b, y2_b = max(0, y1 + y_o), min(h_f, y1 + y_o + f_h)
                                    x1_b, x2_b = max(0, x_c - f_w // 2), min(w_f, x_c + f_w // 2)
                                    if y2_b > y1_b and x2_b > x1_b:
                                        k_size = int(f_w * 0.8) | 1
                                        if k_size < 15: k_size = 15
                                        frame[y1_b:y2_b, x1_b:x2_b] = cv2.GaussianBlur(frame[y1_b:y2_b, x1_b:x2_b], (k_size, k_size), 0)

                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        old_state = self.tracked_objects.get(obj_id, (cx, cy, None))
                        if len(old_state) == 3 and isinstance(old_state[2], bool):
                             old_cx, old_cy, old_zone_id = old_state[0], old_state[1], None
                        else:
                             old_cx, old_cy, old_zone_id = old_state

                        current_zone_id = None 
                        in_any_zone = False
                        for line in self.lines:
                            if line.type == "zone":
                                if line.point_in_polygon((cx, cy)):
                                    in_any_zone = True
                                    current_zone_id = line.zone_id
                                    if old_zone_id != line.zone_id:
                                        now = time.time()
                                        if obj_id not in self.last_seen_time or (now - self.last_seen_time[obj_id]) > anti_double:
                                            self.counts["zone"] += 1     
                                            line.count += 1              
                                            self.last_seen_time[obj_id] = now
                                    break 
                        if not in_any_zone:
                            current_zone_id = None
                        for line in self.lines:
                            if line.type != "zone" and line.is_intersecting((old_cx, old_cy), (cx, cy)):
                                now = time.time()
                                if obj_id not in self.last_seen_time or (now - self.last_seen_time[obj_id]) > anti_double:
                                    if line.type == "entry":
                                        self.counts["entered"] += 1
                                    elif line.type == "exit":
                                        self.counts["exited"] += 1
                                    else:
                                        self.counts["control"] += 1
                                    self.last_seen_time[obj_id] = now
                                    break 
                        self.tracked_objects[obj_id] = (cx, cy, current_zone_id)
                else:
                    if self.face_blur_enabled:
                         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                         min_f = 100 if is_turbo else 50
                         faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_f, min_f))
                         for (fx, fy, fw, fh) in faces:
                             pad = int(fw * 0.1)
                             fx1, fy1 = max(0, fx - pad), max(0, fy - pad)
                             fx2, fy2 = min(frame.shape[1], fx + fw + pad), min(frame.shape[0], fy + fh + pad)
                             face_area = frame[fy1:fy2, fx1:fx2]
                             if face_area.size > 0:
                                 k_size = int(fw * 0.8) | 1
                                 if k_size < 15: k_size = 15
                                 frame[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(face_area, (k_size, k_size), 0)

                self.counts["total"] = total_people
                # Update heatmap before any drawing
                if self.heatmap_enabled or self.save_heatmap:
                    self.update_heatmap(frame.shape, detections)
                frame = self.apply_heatmap(frame)
                current_time = time.time()
                if current_time - last_notification_check > notification_check_interval:
                    self.notification_system.check_thresholds_and_notify(self.counts)
                    last_notification_check = current_time
                
                # Check CSV logging
                if self.save_csv:
                    global_changed = (self.counts["entered"] != last_counts["entered"] or
                                   self.counts["exited"] != last_counts["exited"] or
                                   self.counts["control"] != last_counts["control"] or
                                   self.counts["zone"] != last_counts["zone"] or
                                   self.counts["total"] != last_counts["total"])
                    
                    if global_changed: 
                        row_data = [
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            self.counts["entered"],
                            self.counts["exited"],
                            self.counts["control"],
                            self.counts["zone"],
                            self.counts["total"]
                        ]
                        
                        # Add individual zone counts to row, matching the pre-allocated headers (Zone 1 to 10)
                        current_zones = [line for line in self.lines if line.type == "zone"]
                        for i in range(1, 11): # Zones 1 to 10
                            zone_count = 0
                            # Find zone with this ID or index
                            for z in current_zones:
                                z_id = z.zone_id if z.zone_id else (self.lines.index(z) + 1) # Fallback to index if no ID
                                if z_id == i:
                                    zone_count = z.count
                                    break
                            row_data.append(zone_count)

                        with open(csv_name, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow(row_data)
                        
                        last_counts = self.counts.copy()
                if self.var_debug.get():
                    if results[0].boxes is not None:
                        for box in results[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 1)
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                # ПАКЕТНАЯ ОТРИСОВКА ЗОН (за один проход)
                overlay = frame.copy()
                for line in self.lines:
                    if line.type == "zone" and line.points:
                        pts = np.array(line.points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], True, line.color, 2)
                        cv2.fillPoly(overlay, [pts], line.color)
                
                # Смешиваем все зоны разом
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, dst=frame)
                
                # ПАКЕТНАЯ ОТРИСОВКА ТЕКСТА
                batch_labels = []
                for line in self.lines:
                    if line.type == "zone" and line.points:
                        cx_z = sum(p[0] for p in line.points) // len(line.points)
                        cy_z = sum(p[1] for p in line.points) // len(line.points)
                        batch_labels.append({'text': LANGUAGES[self.lang]["interest_zone"], 'x': cx_z - 50, 'y': cy_z - 10, 'color': line.color})
                    else:
                        cv2.line(frame, line.p1, line.p2, line.color, 3)
                        batch_labels.append({'text': line.type.upper(), 'x': line.p1[0], 'y': line.p1[1]-5, 'color': line.color})
                
                batch_labels += self.get_hud_labels(frame, None)
                frame = self.draw_text_batch(frame, batch_labels)
                if writer:
                    writer.write(frame)
                processed_frames += 1
                progress_percent = (processed_frames / frames_to_process) * 100
                self.root.after(0, lambda p=progress_percent: progress_var.set(p))
                self.root.after(0, lambda p=progress_percent: progress_label.config(text=f"{int(p)}% ({processed_frames}/{frames_to_process} кадров)"))
                progress_window.update()
        finally:
            cap.release()
            if writer:
                writer.release()
            progress_window.destroy()
        # Save heatmap image at the end if enabled
        heatmap_file = None
        if self.save_heatmap:
            heatmap_file = self.save_heatmap_image("file_processing")
            if heatmap_file:
                created_files.append(heatmap_file)
        if self.abort_processing:
            if os.path.exists(video_name):
                os.remove(video_name)
            if self.save_csv and os.path.exists(csv_name):
                os.remove(csv_name)
            self.root.after(0, lambda: messagebox.showinfo(LANGUAGES[self.lang]["success"], LANGUAGES[self.lang]["processing_aborted"]))
        else:
            # Загружаем все файлы в Google Drive
            if self.google_drive.enabled and self.google_drive.folder_id:
                self.upload_all_files_to_google_drive(created_files, "file_processing")
            self.notification_system.check_thresholds_and_notify(self.counts, force=True)
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{hours} {LANGUAGES[self.lang]['hours_unit']} " if hours else ""
            time_str += f"{minutes} {LANGUAGES[self.lang]['mins_unit']} " if minutes or hours else ""
            time_str += f"{seconds} {LANGUAGES[self.lang]['secs_unit']}"
            result_msg = (
                f"{LANGUAGES[self.lang]['processing_complete']}\n"
                f"{LANGUAGES[self.lang]['files_saved_label']}\n"
                f"{LANGUAGES[self.lang]['video_timelapse'].format(video_name)}\n"
            )
            if self.save_csv:
                result_msg += f"{LANGUAGES[self.lang]['csv_report'].format(csv_name)}\n"
            else:
                result_msg += f"{LANGUAGES[self.lang]['csv_disabled']}\n"
            if heatmap_file:
                result_msg += f"{LANGUAGES[self.lang]['heatmap_saved'].format(heatmap_file)}\n"
            if self.google_drive.enabled and self.google_drive.folder_id:
                result_msg += f"\n{LANGUAGES[self.lang]['gdrive_uploaded']}\n"
                folder_url = self.google_drive.get_folder_url(self.google_drive.folder_id)
                if folder_url:
                    result_msg += f"{LANGUAGES[self.lang]['folder_link'].format(folder_url)}\n"
            if self.notification_system.email_enabled:
                result_msg += f"{LANGUAGES[self.lang]['notifications_sent'].format(len(self.notification_system.last_notification_time))}\n"
            result_msg += (
                f"{LANGUAGES[self.lang]['processing_time'].format(time_str)}\n"
                f"{LANGUAGES[self.lang]['frames_processed'].format(processed_frames)}\n"
                f"{LANGUAGES[self.lang]['lines_added'].format(len(self.lines))}\n"
                f"{LANGUAGES[self.lang]['stats'].format(
                    self.counts['entered'],
                    self.counts['exited'],
                    self.counts['control'],
                    self.counts['zone'],
                    self.counts['total']
                )}"
            )
            self.root.after(0, lambda: messagebox.showinfo(LANGUAGES[self.lang]["success"], result_msg))

    def upload_to_google_drive(self, file_path, file_type=""):
        """
        Upload a single file to Google Drive.
        
        Args:
            file_path (str): Path to the file to upload.
            file_type (str): Optional description of the file type.
            
        Returns:
            bool: True if upload was successful, False otherwise.
        """
        if not self.google_drive.enabled or not self.google_drive.folder_id:
            return False
        if not hasattr(self.google_drive, 'service') or not self.google_drive.service:
            if not self.google_drive.authenticate():
                return False
        try:
            success, file_id = self.google_drive.upload_file(file_path, self.google_drive.folder_id)
            if success:
                print(f"✅ Файл '{file_path}' успешно загружен в Google Drive (ID: {file_id})")
                return True
            else:
                print(f"❌ Ошибка загрузки файла '{file_path}': {file_id}")
                return False
        except Exception as e:
            print(f"❌ Критическая ошибка при загрузке '{file_path}': {e}")
            return False

    def upload_all_files_to_google_drive(self, file_paths, session_type=""):
        """
        Upload a list of files to Google Drive.
        
        Args:
             file_paths (list): List of file paths to upload.
             session_type (str): Optional session identifier.
             
        Returns:
            bool: True if all files were uploaded successfully.
        """
        if not self.google_drive.enabled or not self.google_drive.folder_id:
            return False
        if not hasattr(self.google_drive, 'service') or not self.google_drive.service:
            if not self.google_drive.authenticate():
                return False
        success_count = 0
        total_count = len(file_paths)
        for file_path in file_paths:
            if os.path.exists(file_path):
                if self.upload_to_google_drive(file_path):
                    success_count += 1
        print(f"✅ Загружено {success_count} из {total_count} файлов в Google Drive")
        return success_count == total_count

    def create_markup_window(self, first_frame):
        """
        Open a window for drawing lines and zones on the first frame of the video.
        
        Args:
            first_frame (numpy.ndarray): The first frame of the video file.
        """
        self.markup_window = tk.Toplevel(self.root)
        self.markup_window.title(LANGUAGES[self.lang]["markup_window_title"])
        self.theme_manager.apply_to_dialog(self.markup_window)
        self.markup_window.transient(self.root)
        self.markup_window.grab_set()
        self.current_frame = first_frame.copy()
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)))
        self.canvas = tk.Canvas(self.markup_window, width=first_frame.shape[1], height=first_frame.shape[0])
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        control_frame = ttk.Frame(self.markup_window)
        control_frame.pack(fill="x", pady=5)
        self.mode_label = ttk.Label(control_frame, text=LANGUAGES[self.lang]["mode_label"].format(self.current_draw_mode.upper(), len(self.lines)))
        self.mode_label.pack(side="left", padx=10)
        ttk.Button(control_frame, text=LANGUAGES[self.lang]["confirm_button"], command=self.confirm_markup).pack(side="right", padx=5)
        ttk.Button(control_frame, text=LANGUAGES[self.lang]["cancel_button"], command=self.cancel_markup).pack(side="right", padx=5)
        # --- Добавлены элементы управления для тепловой карты в окне разметки ---
        heatmap_control_frame = ttk.Frame(self.markup_window)
        heatmap_control_frame.pack(fill="x", pady=2)
        self.heatmap_label = ttk.Label(heatmap_control_frame, text=LANGUAGES[self.lang]["heatmap_label_on"] if self.heatmap_enabled else LANGUAGES[self.lang]["heatmap_label_off"])
        self.heatmap_label.pack(side="left", padx=10)
        ttk.Button(heatmap_control_frame, text="Toggle (5)", command=self.toggle_heatmap).pack(side="left", padx=5)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.markup_window.bind("<Key>", self.on_key_press)
        self.update_canvas()
        center_window(self.markup_window)

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        if self.current_draw_mode == "zone":
            if not self.drawing_line:
                self.drawing_line.append((x, y))
            else:
                first_point = self.drawing_line[0]
                if math.hypot(x - first_point[0], y - first_point[1]) < 15:
                    if len(self.drawing_line) >= 3:
                        zone_id = len([l for l in self.lines if l.type == "zone"]) + 1
                        zone = CounterLine(None, None, "zone", self.drawing_line.copy(), zone_id=zone_id)
                        zone.calculate_bounds()
                        self.lines.append(zone)
                        self.drawing_line = []
                    else:
                        self.drawing_line.append((x, y))
                else:
                    self.drawing_line.append((x, y))
        else:
            self.drawing_line.append((x, y))
            if len(self.drawing_line) == 2:
                self.lines.append(CounterLine(self.drawing_line[0], self.drawing_line[1], self.current_draw_mode))
                self.drawing_line = []
        self.update_canvas()

    def on_canvas_right_click(self, event):
        if self.lines:
            self.lines.pop()
        elif self.drawing_line:
            self.drawing_line = []
        self.update_canvas()

    def on_key_press(self, event):
        if event.keysym == "Escape":
            self.cancel_markup()
        elif event.keysym == "Return":
            self.confirm_markup()
        elif event.char == "1":
            self.current_draw_mode = "entry"
            self.drawing_line = []
            self.update_mode_label()
        elif event.char == "2":
            self.current_draw_mode = "exit"
            self.drawing_line = []
            self.update_mode_label()
        elif event.char == "3":
            self.current_draw_mode = "control"
            self.drawing_line = []
            self.update_mode_label()
        elif event.char == "4":
            self.current_draw_mode = "zone"
            self.drawing_line = []
            self.update_mode_label()
        elif event.char == "5": # Toggle heatmap
            self.toggle_heatmap()
        elif event.char == "s":
            self.save_lines()
        elif event.char == "l":
            self.load_lines()
        elif event.char == "c":
            self.drawing_line = []
            self.update_canvas()
        elif event.char == "d":
            if self.lines:
                self.lines.pop()
            self.update_canvas()
        elif event.char == "g":
            self.root.after(0, self.open_gdrive_settings)

    def update_canvas(self):
        display_frame = self.current_frame.copy()
        # Apply heatmap if enabled before drawing lines
        if self.heatmap_enabled:
            # For markup window, we can simulate a static heatmap based on current detections or just apply a base effect
            # For simplicity, let's just update the label and apply heatmap when processing video
            pass # Actual heatmap application happens during video processing
        for line in self.lines:
            if line.type == "zone" and line.points:
                pts = np.array(line.points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, line.color, 2)
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [pts], line.color + (50,))
                alpha = 0.2
                display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
                cx = sum(p[0] for p in line.points) // len(line.points)
                cy = sum(p[1] for p in line.points) // len(line.points)
                display_frame = self.draw_text_with_cyrillic(display_frame, LANGUAGES[self.lang]["interest_zone"], cx - 50, cy - 10, line.color)
            else:
                cv2.line(display_frame, line.p1, line.p2, line.color, 3)
                display_frame = self.draw_text_with_cyrillic(display_frame, line.type.upper(), line.p1[0], line.p1[1]-5, line.color)
        if self.current_draw_mode == "zone" and self.drawing_line:
            if len(self.drawing_line) >= 2:
                pts = np.array(self.drawing_line, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], False, (255, 0, 255), 2)
                for point in self.drawing_line:
                    cv2.circle(display_frame, point, 5, (255, 0, 255), -1)
            else:
                if len(self.drawing_line) == 1:
                    cv2.circle(display_frame, self.drawing_line[0], 5, (0, 255, 0), -1)
                elif len(self.drawing_line) == 2:
                    cv2.line(display_frame, self.drawing_line[0], self.drawing_line[1], (0, 255, 255), 2)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
        self.canvas.itemconfig(1, image=self.photo)
        self.update_mode_label()

    def update_mode_label(self):
        self.mode_label.config(text=LANGUAGES[self.lang]["mode_label"].format(self.current_draw_mode.upper(), len(self.lines)))
        self.heatmap_label.config(text=LANGUAGES[self.lang]["heatmap_label_on"] if self.heatmap_enabled else LANGUAGES[self.lang]["heatmap_label_off"])

    def confirm_markup(self):
        self.markup_confirmed = True
        self.markup_window.destroy()
        messagebox.showinfo(LANGUAGES[self.lang]["success"], LANGUAGES[self.lang]["confirm_markup"].format(len(self.lines)))

    def cancel_markup(self):
        self.markup_confirmed = False
        self.markup_window.destroy()

    def get_hud_labels(self, frame, rem):
        """
        Generates a list of labels for the Heads-Up Display (HUD) for batch rendering.
        """
        h, w = frame.shape[:2]
        in_color = (0, 255, 0)
        out_color = (0, 0, 255)
        ctrl_color = (0, 255, 255)
        zone_color = (255, 0, 255)
        total_color = (0, 165, 255)
        white_color = (255, 255, 255)
        
        labels = []
        texts = [
            (LANGUAGES[self.lang]["in_count"].format(self.counts['entered']), in_color),
            (LANGUAGES[self.lang]["out_count"].format(self.counts['exited']), out_color),
            (LANGUAGES[self.lang]["ctrl_count"].format(self.counts['control']), ctrl_color),
            (LANGUAGES[self.lang]["zone_count"].format(self.counts['zone']), zone_color),
            (LANGUAGES[self.lang]["total_count"].format(self.counts['total']), total_color)
        ]
        
        y_pos = 35
        for text, color in texts:
            labels.append({'text': text, 'x': 20, 'y': y_pos, 'color': color})
            y_pos += 35
            
        y_pos += 10
        labels.append({'text': LANGUAGES[self.lang]["zone_table_header"], 'x': 20, 'y': y_pos, 'color': white_color})
        y_pos += 25
        
        header_text = f"{LANGUAGES[self.lang]['zone_id']:<5} {LANGUAGES[self.lang]['count']}"
        labels.append({'text': header_text, 'x': 20, 'y': y_pos, 'color': white_color})
        y_pos += 20
        
        for line in self.lines:
            if line.type == "zone":
                zone_text = f"#{line.zone_id:<4} {line.count}"
                labels.append({'text': zone_text, 'x': 20, 'y': y_pos, 'color': line.color})
                y_pos += 18

        mode_text = LANGUAGES[self.lang]["mode_display"].format(self.current_draw_mode.upper())
        mode_colors = {
            "entry": in_color,
            "exit": out_color,
            "control": ctrl_color,
            "zone": zone_color
        }
        curr_col = mode_colors.get(self.current_draw_mode, white_color)
        y_pos += 10
        labels.append({'text': mode_text, 'x': 20, 'y': y_pos, 'color': curr_col})
        
        curr_time_str = datetime.now().strftime("%H:%M:%S")
        time_x = w - 180
        time_text = LANGUAGES[self.lang]["current_time"].format(curr_time_str)
        labels.append({'text': time_text, 'x': time_x, 'y': 35, 'color': white_color})
        
        fps_text = f"FPS: {self.fps:.1f}"
        labels.append({'text': fps_text, 'x': time_x, 'y': 70, 'color': white_color})
        
        return labels

    
if __name__ == "__main__":
    PeopleCounterPRO()  


#zxc

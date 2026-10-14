# PLAN — Hand Portal для Deep-Live-Cam

## Цель

Добавить эффект «портал из рук» по референсу (`tmp/v14044g50000da8udo7og65jntmuv5ng.mov`):
две руки в кадре задают четырёхугольник по кончикам больших (landmark 4) и
указательных (landmark 8) пальцев. Внутри него поверх кадра применяется
фильтр, снаружи кадр не меняется, контур рисуется тонкой белой линией.
Самопересекающийся контур («бабочка») допустим, как в референсе.
Щепоть одной руки (все 5 кончиков пальцев сведены вместе) переключает на
следующий фильтр.

## Решения (согласованы 2026-10-14)

- Детектор рук: **MediaPipe** `HandLandmarker` (Tasks API, `mediapipe==1.1.0`),
  модель `hand_landmarker.task` (7 819 105 байт) с
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
  скачивается через `modules/model_downloader.py` в `models/`.
- MediaPipe тянет `opencv-contrib-python`, поэтому в `requirements.txt`
  `opencv-python==4.14.0.94` → `opencv-contrib-python==4.14.0.94`
  (`opencv-python-headless` остаётся как есть).
- Режимы: **живое превью + обработка image/video** (и headless через
  `--frame-processor hand_portal`).
- Управление: **жест-щепоть + UI** (переключатель «Hand Portal» и выпадающий
  список стартового фильтра).

## Архитектура

- `modules/hand_portal.py` — **вся логика**: реестр фильтров, геометрия
  портала, детекция щепоти, обёртка над MediaPipe, публичная функция
  `apply_hand_portal(frame) -> frame`.
- `modules/processors/frame/hand_portal.py` — тонкий адаптер под интерфейс
  фрейм-процессора (`pre_check`, `pre_start`, `process_frame`,
  `process_frame_v2`, `process_frames`, `process_image`, `process_video`),
  без собственной логики, только вызовы `modules.hand_portal`.
- Регистрация: `ALLOWED_PROCESSORS` в `modules/processors/frame/core.py`,
  `choices` у `--frame-processor` в `modules/core.py`, ключ
  `fp_ui["hand_portal"]` в `modules/globals.py`.
- `modules/ui.py` — только точки интеграции: переключатель, выпадающий список
  фильтров, сохранение состояния. В живом превью процессор вызывается
  существующей общей веткой `fp.process_frame(...)` / `process_frame_v2(...)`,
  после face swap, потому что модуль добавляется в конец конвейера.
- Состояние в `modules/globals.py`: `hand_portal: bool = False`,
  `hand_portal_filter: str = "purple"`.
- Потоки: обработка видео идёт в `ThreadPoolExecutor`, поэтому у каждого потока
  свой экземпляр `HandLandmarker` (`threading.local`), режим `IMAGE`.
  Переключение фильтра жестом работает только в живом превью
  (`modules.globals.webcam_preview_running`), в файлах фильтр фиксирован.

## Стек

Python 3.10+ (локально `.venv` на Python 3.12), PySide6, OpenCV
(contrib), NumPy 2, MediaPipe 1.1.0, onnxruntime, pytest, ruff.

## Подпланы

| #  | Файл                     | Содержание                     | Статус |
| -- | ------------------------ | ------------------------------ | ------ |
| 01 | `plan/01-hand-portal.md` | Hand Portal: от каркаса до UI  | todo   |

## Проверки перед мержем

```
python -m pytest tests -q
ruff check .
python -m compileall -q modules run.py
```

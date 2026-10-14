# 01 — Hand Portal

Статус подплана: **in_progress** (ок получен 2026-10-14).
Run Orca: `run_9e29e1247d44`

Исполнитель `haiku` = `--agent claude --model claude-haiku-5-5` (Haiku 5.5).

Общие требования ко всем задачам (включаются в каждый spec):
- Не ломать существующие режимы: image/video processing, headless, live preview.
- Стиль проекта: type hints, docstrings, `NAME = "DLC.…"` в процессорах, без
  новых линтер-ошибок (`ruff check .`).
- Работать только в своём worktree, закоммитить в своей ветке, не пушить.
- Проверки: `python -m pytest tests -q`, `ruff check .`,
  `python -m compileall -q modules run.py` (`.venv/bin/python`).

Подготовка координатора (не задача): создать `.venv` (Python 3.12) в основном
checkout и поставить `requirements.txt`, чтобы `.venv` подключался в
worktree через `orca.yaml`.

## Граф

```
01-01 ──► 01-02 ──┬──► 01-03 (processor + CLI)
                  ├──► 01-04 (UI)
                  ├──► 01-05 (доп. фильтры)
                  └──► 01-06 (README + локали)
```

---

### 01-01 — Зависимости и загрузка модели

- **Исполнитель:** haiku · **Статус:** todo · **Зависимости:** — · **Orca:** `task_c1bf43d9ebd0`
- **Файлы:** `requirements.txt`, `modules/model_downloader.py`,
  `tests/test_model_downloader_external_url.py` (новый)
- **Описание:**
  - `requirements.txt`: `opencv-python==4.14.0.94` →
    `opencv-contrib-python==4.14.0.94`; добавить `mediapipe==1.1.0`.
    Остальные строки не трогать.
  - `model_downloader.py`: добавить словарь `MODEL_URLS: Dict[str, str]`
    (внешние URL для моделей не из HF-репозитория); `resolve_url(name)`
    возвращает `MODEL_URLS[name]`, если имя там есть, иначе прежнее поведение.
    Добавить `"hand_landmarker.task"` в `MODEL_SIZES` (7819105) и в
    `MODEL_URLS` (URL из PLAN.md). TLS-проверку не ослаблять.
- **Критерии готовности:** тест проверяет, что `resolve_url("hand_landmarker.task")`
  отдаёт URL Google Storage, а `resolve_url("inswapper_128.onnx")` — прежний
  HF URL; `expected_size("hand_landmarker.task") == 7819105`; все проверки зелёные.

### 01-02 — Ядро `modules/hand_portal.py`

- **Исполнитель:** haiku · **Статус:** todo · **Зависимости:** 01-01 · **Orca:** `task_be6c033bc45d`
- **Файлы:** `modules/hand_portal.py` (новый), `modules/globals.py`,
  `tests/test_hand_portal.py` (новый)
- **Описание:** вся логика портала в одном модуле. Публичный API (имена
  фиксированы, на них опираются 01-03…01-06):
  - `Point = Tuple[float, float]`, `Hand = List[Point]` (21 landmark MediaPipe
    в пиксельных координатах), `FilterFn = Callable[[np.ndarray], np.ndarray]`.
  - `FILTERS: Dict[str, FilterFn]` (порядок вставки = порядок переключения),
    `register_filter(name: str, fn: FilterFn) -> None`,
    `filter_names() -> List[str]`, `next_filter(name: str) -> str` (по кругу;
    неизвестное имя → первый фильтр).
  - Базовые фильтры: `purple` (смешивание с фиолетовым ~0.6), `grayscale`,
    `negative`. Фильтр принимает и возвращает BGR uint8 того же размера.
  - `portal_corners(hands: Sequence[Hand]) -> Optional[np.ndarray]`: `None`, если
    рук меньше двух; иначе берёт две руки, упорядочивает по x запястья
    (landmark 0) слева направо и возвращает `int32` массив 4×2:
    `[A.index_tip(8), B.index_tip(8), B.thumb_tip(4), A.thumb_tip(4)]`.
  - `apply_filter_in_polygon(frame, polygon, filter_name) -> np.ndarray`:
    маска через `cv2.fillPoly` (самопересечение допустимо), фильтр применяется
    только к bbox полигона ради скорости, белый контур `cv2.polylines` толщиной 2
    (antialiased). Пустой/вырожденный полигон или неизвестный фильтр → кадр без
    изменений. Входной кадр не мутировать.
  - `is_pinch(hand: Hand, ratio: float = 0.25) -> bool`: все кончики
    (4, 8, 12, 16, 20) ближе `ratio * hand_size` к их центроиду, где
    `hand_size = dist(landmark 0, landmark 9)`.
  - `class PinchCycler`: `update(hands: Sequence[Hand]) -> bool` — True только
    на фронте (щепоть появилась у любой руки после кадра без щепоти) и не чаще
    одного раза в `cooldown_s=0.8` секунды (время через `time.monotonic`,
    подменяемое в тестах параметром конструктора `clock`).
  - `ensure_model() -> Optional[str]` — путь к `hand_landmarker.task` через
    `model_downloader.ensure_model`.
  - `detect_hands(frame: np.ndarray) -> List[Hand]`: MediaPipe Tasks
    `HandLandmarker`, режим IMAGE, `num_hands=2`, ленивый импорт mediapipe и
    отдельный экземпляр на поток (`threading.local`). Нет модели или mediapipe →
    одно предупреждение в консоль и `[]`, без исключений.
  - `apply_hand_portal(frame: np.ndarray, hands: Optional[Sequence[Hand]] = None) -> np.ndarray`:
    если `hands is None` — детектирует; при `modules.globals.webcam_preview_running`
    прогоняет `PinchCycler` (модульный экземпляр) и при срабатывании пишет
    `next_filter(...)` в `modules.globals.hand_portal_filter`; затем рисует портал
    фильтром `modules.globals.hand_portal_filter`.
  - `modules/globals.py`: `hand_portal: bool = False`,
    `hand_portal_filter: str = "purple"`, ключ `"hand_portal": False` в `fp_ui`.
- **Критерии готовности:** тесты без mediapipe и без камеры, на синтетических
  руках: порядок углов в `portal_corners`; `None` при одной руке; пиксели внутри
  полигона изменены, снаружи совпадают с оригиналом; вход не мутирован;
  `is_pinch` true/false; `PinchCycler` срабатывает один раз на удержание и
  уважает cooldown; `next_filter` по кругу; `detect_hands` возвращает `[]`, если
  импорт mediapipe замокан на ошибку. Все проверки зелёные.

### 01-03 — Фрейм-процессор и CLI

- **Исполнитель:** haiku · **Статус:** todo · **Зависимости:** 01-02 · **Orca:** `task_258d6022d063`
- **Файлы:** `modules/processors/frame/hand_portal.py` (новый),
  `modules/processors/frame/core.py` (только `ALLOWED_PROCESSORS`),
  `modules/core.py` (только argparse и разбор аргументов),
  `tests/test_hand_portal_processor.py` (новый)
- **Описание:** тонкий адаптер без логики: `NAME = "DLC.HAND-PORTAL"`;
  `pre_check()` вызывает `hand_portal.ensure_model()` (False + `update_status`
  при неудаче); `pre_start()` → True; `process_frame(source_face, temp_frame, **kwargs)`
  и `process_frame_v2(temp_frame, **kwargs)` → `apply_hand_portal(temp_frame)`;
  `process_frames`, `process_image`, `process_video` — по образцу
  `face_enhancer.py` (чтение/запись через существующие утилиты, `multi_process_frame`).
  Добавить `'hand_portal'` в `ALLOWED_PROCESSORS` и в `choices` у
  `--frame-processor`; новый аргумент `--hand-portal-filter` (по умолчанию
  `purple`) пишет в `modules.globals.hand_portal_filter`; при наличии
  `hand_portal` в `--frame-processor` ставить `modules.globals.hand_portal = True`
  и `fp_ui["hand_portal"] = True`. Поведение без этих флагов не меняется.
- **Критерии готовности:** тест: модуль проходит проверку
  `load_frame_processor_module("hand_portal")` (все методы
  `FRAME_PROCESSORS_INTERFACE` есть); `process_frame(None, frame, target_face=None)`
  вызывает `apply_hand_portal` (замокан) и возвращает его результат. Все проверки зелёные.

### 01-04 — Интеграция в UI

- **Исполнитель:** haiku · **Статус:** todo · **Зависимости:** 01-02 · **Orca:** `task_3fb88cfaafa4`
- **Файлы:** `modules/ui.py` (только точки интеграции)
- **Описание:** в `_build_options_card` добавить переключатель `_Switch`
  «Hand Portal» (tooltip: «Frame a filter portal between both hands; pinch all
  fingers to switch filter») и `QComboBox` со значениями
  `hand_portal.filter_names()` (текущий — `modules.globals.hand_portal_filter`).
  Переключатель пишет `modules.globals.hand_portal` и
  `modules.globals.fp_ui["hand_portal"]`, комбобокс — `hand_portal_filter`;
  оба вызывают `save_switch_states()`. В `save_switch_states`/`load_switch_states`
  сохранить и загрузить `hand_portal` и `hand_portal_filter` (с дефолтами при
  отсутствии ключей). Если фильтр переключили жестом, комбобокс подтягивает
  значение не реже раза в секунду (`QTimer`, только когда значение изменилось).
  Сам вызов портала в `_ProcessingThread` не добавлять: процессор `hand_portal`
  подхватывается существующим `get_frame_processors_modules` через `fp_ui`.
  Строки UI оборачивать в `_()`.
- **Критерии готовности:** `python -c "import modules.ui"` без ошибок;
  `compileall`, `ruff`, существующие тесты зелёные; diff `ui.py` — только
  перечисленные точки.

### 01-05 — Дополнительные фильтры

- **Исполнитель:** antigravity · **Статус:** todo · **Зависимости:** 01-02 · **Orca:** `task_1852cd8197da`
- **Файлы:** `modules/hand_portal.py` (только секция фильтров, через
  `register_filter`/словарь `FILTERS`), `tests/test_hand_portal_filters.py` (новый)
- **Описание:** добавить фильтры из референса после базовых, в этом порядке:
  `sketch` (карандашный набросок: grayscale → invert → GaussianBlur → color dodge),
  `thermal` (`cv2.applyColorMap(..., cv2.COLORMAP_JET)`), `sepia` (матрица
  сепии), `pixelate` (уменьшение до 1/12 и обратно `INTER_NEAREST`), `cool`
  (сдвиг в сине-бирюзовый: усиление B/G, ослабление R). Остальной код
  модуля не трогать.
- **Критерии готовности:** тест: для каждого фильтра выход той же формы и
  dtype `uint8`, что вход (включая кадры 1×1 и 7×5); `filter_names()` начинается
  с `purple, grayscale, negative, sketch, thermal, sepia, pixelate, cool`.
  Все проверки зелёные.

### 01-06 — Документация и локали

- **Исполнитель:** antigravity · **Статус:** todo · **Зависимости:** 01-02 · **Orca:** `task_d83bdd65b867`
- **Файлы:** `README.md`, `locales/*`
- **Описание:** раздел «Hand Portal» в README: что делает, как включить в UI,
  жест-щепоть, список фильтров, CLI-пример
  `python run.py -s face.jpg -t video.mp4 -o out.mp4 --frame-processor face_swapper hand_portal --hand-portal-filter thermal`,
  что модель скачивается автоматически. В файлах `locales/` добавить переводы
  новых строк UI: «Hand Portal», «Portal filter:» и tooltip из 01-04 (формат —
  как у существующих ключей; для языков, которые не знаешь, ключ = английский
  текст).
- **Критерии готовности:** README рендерится без битой разметки; все файлы
  локалей валидны в своём формате (JSON парсится, если JSON). Проверки зелёные.

---

## Финальная проверка координатора

После мержа всех задач: полный прогон проверок в `main`, затем прошу тебя
проверить живое превью с камерой (две руки → портал, щепоть → смена фильтра)
и обработку одного короткого видео.

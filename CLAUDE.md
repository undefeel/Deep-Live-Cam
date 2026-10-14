# CLAUDE.md — координатор проекта Deep-Live-Cam

## Роль

Ты — координатор проекта на модели Opus. Код сам не пишешь (кроме мелких
правок при мерже): планируешь, раздаёшь задачи агентам через Orca,
проверяешь и мержишь.

## Исполнители

| Исполнитель   | Команда запуска                                                            | Для чего                                                     |
| ------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `haiku`       | `--agent claude --model claude-haiku-5-5`                                             | логика, бэкенд, тесты                                        |
| `antigravity` | `--agent antigravity` (без `--model`, модель из настроек Antigravity)      | простые изолированные задачи, вёрстка, документация, конфиги |

- ID модели Haiku: `claude-haiku-5-5` (Haiku 5.5, явный ID по требованию пользователя; алиас `haiku` не использовать).
- Вместо Gemini CLI используется Antigravity (`agy`): `gemini` в системе не
  установлен. Везде, где говорится «gemini», имеется в виду `antigravity`.
- После запуска сверяй `launch.requested` с `launch.effective` в ответе
  `worker-start`. По одним аргументам запроса нельзя утверждать, какая модель
  на самом деле запустилась.

## Структура плана

- `plan/PLAN.md` — архитектура, стек, список подпланов (этапов) и их порядок.
- `plan/NN-<название>.md` — подплан этапа с задачами.
- Каждая задача: ID (например `02-03`), описание, критерии готовности,
  зависимости, затрагиваемые файлы/модули, исполнитель (`haiku` | `antigravity`),
  статус (`todo` | `in_progress` | `review` | `done` | `failed`).
- Задачи маленькие и самодостаточные: исполнитель видит только текст задачи,
  поэтому в нём должно быть всё нужное. Spec задачи для Orca обязательно
  содержит: **Target** (файлы/модуль), **Change** (что сделать),
  **Constraints** (что нельзя трогать, инварианты), **Ownership** (что этот
  исполнитель может править), **Observable acceptance** (какой тест/вывод
  доказывает готовность). В spec указывай: закоммить изменения в своей ветке
  worktree и не пушить.
- Планируй под параллельность: независимые задачи не трогают одни и те же
  файлы. Каркас проекта — первая задача, идёт одна.
- Выбор исполнителя: `haiku` — логика, бэкенд, тесты; `antigravity` — простые
  изолированные задачи, вёрстка, документация, конфиги. Если задача слишком
  сложна для обоих — дроби её дальше.

## Порядок работы

1. Новая идея → сначала уточняющие вопросы, потом план. Закоммить `plan/`
   и жди моего «ок». Без «ок» к выполнению не переходи.
2. Выполнение: создай Run, заведи задачи текущего подплана с зависимостями:
   ```
   orca orchestration run-create --objective "<подплан NN>" --json
   orca orchestration task-create --spec "<spec>" --deps '<json_array task_id>' --json
   orca orchestration task-list --ready --brief --json
   ```
   Соответствие «ID задачи в plan/ ↔ `task_id` Orca» записывай в подплан.
3. Все задачи с выполненными зависимостями запускай одновременно, каждую
   в новом дочернем worktree:
   ```
   orca orchestration worker-start --task <task_id> --worktree new-child --name <NN-MM-slug> --agent claude --model claude-haiku-5-5 --json
   orca orchestration worker-start --task <task_id> --worktree new-child --name <NN-MM-slug> --agent antigravity --json
   ```
   Одновременно не больше 4 исполнителей. Если `worker-start` завершился
   с ошибкой — не перезапускай вслепую: смотри `failedStage` и
   `residualResources` и следуй `orca skills get orchestration --reference references/recovery-and-cleanup.md`.
4. Жди worker_done:
   ```
   orca orchestration check --wait --types "worker_done,escalation,question" --timeout-ms 900000 --json
   ```
   Таймаут или пустой ответ — это контрольная точка, а не провал. На
   `question` отвечай через `orca orchestration reply --id <message_id> --body "..." --json`.
   После трёх пустых ожиданий подряд смотри
   `orca orchestration worker-list --run <run_id> --include-remote --json`
   и действуй по `projection.nextAction`.
   По каждой задаче проверь diff и запусти тесты:
   - хорошо → мержи в main, статус `done`;
   - плохо → верни на доработку с конкретными замечаниями (новый Dispatch
     той же задачи: `worker-start --task <task_id> --retry-of <dispatch_id> --worktree new-child ...`
     или, если воркер ещё не освобождён, `worker-start --task <task_id> --terminal <handle>`);
   - после двух неудач → перезапусти на другом исполнителе или раздели задачу.
     Учти: после трёх проваленных Dispatch Orca сама переводит задачу в
     `failed` — дальше только новая задача с изменённым spec.
5. Мержи по одной задаче, после каждого мержа прогоняй тесты в main
   (см. «Мерж и тесты»).
6. После мержа запускай разблокированные задачи. Отработавшие воркеры
   освобождай: `orca orchestration worker-release --dispatch <dispatch_id> --json`
   (только после принятого worker_done), затем подтверждай доставку:
   `orca orchestration check --ack <delivery_id> --json`. Смёрженные
   worktree удаляй: `orca worktree rm --worktree "path:<путь>" --json`.
   Перед закрытием Run команда
   `orca orchestration worker-list --run <run_id> --terminal-state reclaimable --json`
   должна возвращать пустой список.
7. Подплан закрыт → короткий отчёт мне: что сделано, что не вышло, что
   дальше. Затем переходи к следующему подплану без ожидания, если я не
   сказал иначе.
8. Перед каждым ответом мне держи `plan/` в актуальном состоянии — это
   единственный источник правды о прогрессе.

Валидный worker_done сам закрывает Task и Dispatch — не вызывай после него
`task-update --status completed`. Не запускай `orca orchestration reset`:
он сбрасывает состояние глобально, включая Run'ы других проектов.

## Мерж и тесты

- Никогда не `git add -A` / `git commit -a` с новыми файлами в worktree: Orca кладёт
  туда симлинк `.venv`, и при мерже он заменяет настоящее окружение в основном
  checkout. Добавляй файлы явно и проверяй `git show --stat` перед мержем.

- Основной checkout с веткой `main`: `/Users/kodoku/orca/Deep-Live-Cam`.
  Рабочий worktree координатора — ветка `undefeel/core`, от неё ветвятся
  дочерние worktree исполнителей.
- Мерж задачи:
  ```
  git -C /Users/kodoku/orca/Deep-Live-Cam merge --no-ff <ветка_исполнителя> -m "merge <ID задачи>: <кратко>"
  git -C /Users/kodoku/orca/Deep-Live-Cam push origin main
  git merge --ff-only main          # в worktree координатора, чтобы новые дочерние worktree стартовали от свежего main
  ```
- Тесты и проверки (как в CI):
  ```
  python -m pytest tests -q
  ruff check .
  python -m compileall -q modules run.py
  ```

## Если сессия прервалась

Восстанови состояние по `plan/` и
`orca orchestration run-list --json` → `orca orchestration run-use --id <run_id> --json` →
`orca orchestration task-list --brief --json`, плюс
`orca orchestration worker-list --run <run_id> --include-remote --json`.
Затем продолжай с того места, где остановился. Актуальные команды и флаги
сверяй с `orca skills get orchestration --full`.

## Остановись и спроси меня, если

- нужно менять архитектуру или стек из `PLAN.md`;
- нужны секреты, ключи, платные сервисы;
- задача проваливается после перезапуска на другом исполнителе.

# Safetensors Inference Rig

RunPod-панель для запуска Hugging Face моделей в формате Safetensors через vLLM. Один и тот же checkpoint можно активировать с вычислениями `BF16`, `FP16` или `FP32`; API совместим с OpenAI.

## Возможности

- поиск локальных моделей в `/workspace/models/safetensors/<org>/<repo>`;
- проверка Hugging Face-репозитория и скачивание полного snapshot: `.safetensors`, `config.json`, tokenizer и model code;
- альтернативные веса `.bin`, `.gguf`, ONNX и другие тяжёлые форматы при скачивании пропускаются;
- выбор `BF16`, `FP16` или `FP32` при активации;
- настройки context length, concurrent sequences, tensor parallel и доли GPU memory;
- один управляемый vLLM server с health check, graceful shutdown, сохранением активной конфигурации и rollback;
- OpenAI-compatible API с Bearer key и Gradio с Basic Auth;
- **удаление моделей** прямо из панели;
- **Chat Playground** с настройками генерации: temperature, max tokens, top-p, repetition penalty, system prompt;
- **vLLM метрики** на дашборде: active/pending requests, KV cache usage, tokens/sec;
- **API статистика**: счётчик запросов, токенов, средняя латентность;
- **автоматический перезапуск** при краше vLLM (с exponential backoff);
- **ротация логов** vLLM при превышении лимита размера.
- выбор Hugging Face branch/tag/commit с фиксацией фактического commit SHA;
- staging-статусы и отмена загрузки;
- GPU topology/VRAM preflight до остановки текущей модели;
- автоматическое уменьшение context length после KV-cache/CUDA OOM;
- persistent JSONL-журнал и история запусков/ошибок;
- публичный panel health probe `GET /healthz`.
- аудио-ввод в Playground и OpenAI-compatible `audio_url` для моделей вроде Audio Flamingo Next.

> Выбор dtype задаёт вычислительный dtype vLLM. Он не переписывает исходные `.safetensors` на диске. FP32 обычно требует примерно вдвое больше памяти весов, чем BF16/FP16. BF16 требует совместимую GPU (обычно NVIDIA Ampere или новее).

## RunPod

Используйте актуальный CUDA 13 / vLLM template, network volume с mount path `/workspace` и откройте HTTP-порты `7860,8000`. Для Audio Flamingo Next нужен `vllm==0.20.0` и Transformers 5.6+; старый CUDA 12.8 / Torch 2.8 bootstrap для этой модели не подходит. При собственном Docker image ориентируйтесь на `vllm/vllm-openai:v0.20.0` и совместимый NVIDIA driver. Проект запускает vLLM через небольшой compatibility launcher: он вычисляет отсутствующие `rote_timestamps` для опубликованного процессора Audio Flamingo Next, без изменения установленного пакета vLLM.

Secrets / Environment Variables:

```text
SAFETENSORS_API_KEY=<случайный длинный токен>
SAFETENSORS_PANEL_USER=<логин>
SAFETENSORS_PANEL_PASSWORD=<сложный пароль>
HF_TOKEN=<необязательно; нужен для gated/private моделей>
```

Команду запуска возьмите из `runpod_command.txt`. Она также включает первый bootstrap: скачивает `nvidia/audio-flamingo-next-hf`, запускает его в BF16 и сохраняет конфигурацию на volume. После этого Pod восстанавливает уже скачанную модель, без повторной загрузки. Первый запуск создаёт persistent venv. Последующие запуски пропускают установку, пока requirements, constraints и параметры установки не изменились. Для Audio Flamingo Next bootstrap устанавливает pinned `vllm==0.20.0` и Transformers 5.6+. Он намеренно не фиксирует Torch: vLLM должен выбрать wheel, соответствующий CUDA/Python в вашем RunPod template.

После readiness в логах Pod и на dashboard появится готовый OpenAI-compatible адрес:

```text
https://<RUNPOD_POD_ID>-8000.proxy.runpod.net/v1
```

Используйте `SAFETENSORS_API_KEY` как Bearer token. URL строится автоматически из переменной RunPod `RUNPOD_POD_ID`; при custom domain его можно переопределить через `SAFETENSORS_PUBLIC_API_URL`. Панель доступна по `https://<RUNPOD_POD_ID>-7860.proxy.runpod.net`.

Если хотите запретить установку vLLM через pip совсем, задайте `SAFETENSORS_INSTALL_VLLM=0`; запуск остановится с ошибкой, если `vllm` не импортируется.

## Работа в панели

1. Введите `organization/repository`, нужную revision (`main`, branch, tag или commit) и нажмите **Inspect repository**. Панель покажет разрешённый commit SHA.
2. Нажмите **Download snapshot**. Файлы сначала загружаются в staging-каталог; модель появляется в списке только после проверки `config.json`, Safetensors index и полного набора шардов.
3. Выберите dtype и параметры запуска, затем **Activate model**.
4. Для репозиториев с собственным Python-кодом включайте `Trust repository remote code` только если доверяете автору.
5. Для удаления модели выберите её в списке, отметьте подтверждение и нажмите **Delete selected model**.

Перед переключением выполняется проверка числа GPU и приблизительной памяти весов на GPU. Если startup не помещает KV cache или получает CUDA OOM, включённый по умолчанию `Auto-reduce context` повторяет запуск с вдвое меньшим context length, но не ниже 512.

### Chat Playground

В разделе **Playground** можно тестировать модель прямо в панели. В настройках генерации доступны:

- **System prompt** — системное сообщение
- **Temperature** (0.0–2.0) — креативность генерации
- **Max tokens** (1–16384) — максимальная длина ответа
- **Top-p** (0.0–1.0) — nucleus sampling
- **Repetition penalty** (1.0–2.0) — штраф за повторы

Если ответ обрезан по лимиту max_tokens, в конце сообщения появится предупреждение.
Playground запрашивает usage в streaming-ответе и показывает точные completion tokens/tokens per second. Для старых API без streaming usage явно показывается число chunks, а не приблизительное число токенов.

### Audio Flamingo Next

Загрузите `nvidia/audio-flamingo-next-hf`, выберите BF16 и активируйте его с `Max audio clips per request = 1`. В Playground загрузите WAV/MP3/OGG/FLAC/M4A/WEBM и сформулируйте задачу явно, например: `Transcribe the input speech.` или `Generate a caption for the input audio.` Модель ожидает mono 16 kHz аудио; внутренне она работает 30-секундными окнами и рассчитана на длинные записи.

Playground передаёт файл как `audio_url` data URL в OpenAI-compatible `POST /v1/chat/completions`. Эквивалентный API-запрос:

```json
{
  "model": "current",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Transcribe the input speech."},
      {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,<BASE64_AUDIO>"}}
    ]
  }]
}
```

### Дашборд

Дашборд отображает:

- Статус vLLM, модель, PID, uptime
- Параметры активной модели (context length, TP, GPU utilization и т.д.)
- GPU: загрузка, память, температура (обновляется каждые 5 секунд, данные кэшируются на 3 секунды)
- **vLLM метрики**: active/pending requests, KV cache usage %, tokens/sec
- **API статистика**: общее число запросов, токенов, средняя латентность, ошибки
- Количество скачанных моделей на volume

Запрос к API:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $SAFETENSORS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"current","messages":[{"role":"user","content":"Привет"}]}'
```

## Локальная разработка

Backend-тесты не требуют GPU или установленного vLLM:

```bash
python3 -m venv .venv
.venv/bin/pip install pytest huggingface-hub
PYTHONPATH=. .venv/bin/python -m pytest -q
```

Для полного запуска установите `requirements.txt` и привяжите оба listener к localhost:

```bash
SAFETENSORS_ALLOW_INSECURE=1 \
SAFETENSORS_API_HOST=127.0.0.1 \
SAFETENSORS_PANEL_HOST=127.0.0.1 \
.venv/bin/python app.py
```

## Переменные

| Variable | Default | Описание |
| --- | --- | --- |
| `SAFETENSORS_VOLUME_ROOT` | `/workspace` | Корень persistent volume |
| `SAFETENSORS_MODELS_DIR` | `/workspace/models/safetensors` | Директория моделей |
| `SAFETENSORS_STATE_DIR` | `/workspace/.state/safetensors-rig` | Директория состояния |
| `SAFETENSORS_LOG_DIR` | `/workspace/logs/safetensors-rig` | Директория логов |
| `VLLM_PYTHON` | текущий Python interpreter | Путь к Python для vLLM |
| `SAFETENSORS_API_PORT` | `8000` | Порт OpenAI API |
| `SAFETENSORS_PANEL_PORT` | `7860` | Порт Gradio панели |
| `SAFETENSORS_PANEL_FIND_FREE_PORT` | `0` | Искать другой panel port, если заданный занят; включайте только локально |
| `SAFETENSORS_BOOTSTRAP_MODEL` | пусто | Модель, которую скачать и запустить при первом старте без сохранённой активной модели. `runpod_command.txt` задаёт `nvidia/audio-flamingo-next-hf` |
| `SAFETENSORS_BOOTSTRAP_REVISION` | `main` | Branch/tag/commit для первичной загрузки |
| `SAFETENSORS_BOOTSTRAP_DTYPE` | `bfloat16` | dtype первого автоматического запуска |
| `SAFETENSORS_BOOTSTRAP_MAX_MODEL_LEN` | `8192` | Context length первого запуска |
| `SAFETENSORS_BOOTSTRAP_MAX_NUM_SEQS` | `64` | Concurrent sequences первого запуска |
| `SAFETENSORS_BOOTSTRAP_TENSOR_PARALLEL_SIZE` | `1` | Tensor parallel для первого запуска |
| `SAFETENSORS_BOOTSTRAP_GPU_MEMORY_UTILIZATION` | `0.90` | Доля памяти GPU для первого запуска |
| `SAFETENSORS_BOOTSTRAP_MAX_AUDIO_PER_PROMPT` | `1` | Максимум audio clips в одном запросе для первого запуска |
| `SAFETENSORS_PUBLIC_API_URL` | автоматически на RunPod | Явный public API URL для custom domain; иначе `https://<pod>-8000.proxy.runpod.net` |
| `SAFETENSORS_PUBLIC_PANEL_URL` | автоматически на RunPod | Явный public panel URL для custom domain; иначе `https://<pod>-7860.proxy.runpod.net` |
| `SAFETENSORS_HEALTH_TIMEOUT` | `600` секунд | Таймаут health check при старте |
| `SAFETENSORS_STOP_TIMEOUT` | `30` секунд | Таймаут graceful shutdown |
| `SAFETENSORS_AUTO_RESTART` | `0` (выключено) | Автоматический перезапуск vLLM при краше |
| `SAFETENSORS_AUTO_RESTART_MAX_RETRIES` | `3` | Максимум попыток авто-рестарта |
| `SAFETENSORS_MAX_LOG_BYTES` | `52428800` (50 MB) | Лимит размера vllm.log перед ротацией |
| `SAFETENSORS_MAX_AUDIO_UPLOAD_BYTES` | `104857600` (100 MB) | Лимит audio upload в Playground |

Structured lifecycle/runtime events сохраняются в `/workspace/logs/safetensors-rig/events.jsonl`; предыдущая ротация — `events.jsonl.1`. Для RunPod health check используйте `http://<panel>:7860/healthz`.

## Проверки качества

Репозиторий содержит GitHub Actions workflow для `pytest`, Ruff, Mypy и ShellCheck. Локально:

```bash
python -m pip install pytest huggingface-hub tqdm ruff mypy
PYTHONPATH=. python -m pytest -q
ruff check .
mypy
shellcheck start_runpod.sh install_runpod.sh generate_env.sh
```

Старые `GGUF_*` имена читаются как fallback для API/panel/paths, но для новых deployments используйте `SAFETENSORS_*`.

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

> Выбор dtype задаёт вычислительный dtype vLLM. Он не переписывает исходные `.safetensors` на диске. FP32 обычно требует примерно вдвое больше памяти весов, чем BF16/FP16. BF16 требует совместимую GPU (обычно NVIDIA Ampere или новее).

## RunPod

Используйте актуальный CUDA/PyTorch Ubuntu template, network volume с mount path `/workspace` и откройте HTTP-порты `7860,8000`. Проверенный вариант под этот bootstrap: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

Secrets / Environment Variables:

```text
SAFETENSORS_API_KEY=<случайный длинный токен>
SAFETENSORS_PANEL_USER=<логин>
SAFETENSORS_PANEL_PASSWORD=<сложный пароль>
HF_TOKEN=<необязательно; нужен для gated/private моделей>
```

Команду запуска возьмите из `runpod_command.txt`. Первый запуск создаёт persistent venv. По умолчанию venv создаётся с `--system-site-packages`, поэтому он видит пакеты из RunPod template. Если vLLM не установлен в template, bootstrap ставит pinned `vllm==0.11.0` через constraints под `torch==2.8.0`, чтобы pip не выбрал новую vLLM и не потянул PyTorch/CUDA/NVIDIA стек другой версии.

Если хотите запретить установку vLLM через pip совсем, задайте `SAFETENSORS_INSTALL_VLLM=0`; запуск остановится с ошибкой, если `vllm` не импортируется.

## Работа в панели

1. Введите `organization/repository` и нажмите **Inspect repository**.
2. Нажмите **Download snapshot**. Модель появится в локальном списке только после наличия `config.json` и хотя бы одного `.safetensors`.
3. Выберите dtype и параметры запуска, затем **Activate model**.
4. Для репозиториев с собственным Python-кодом включайте `Trust repository remote code` только если доверяете автору.
5. Для удаления модели выберите её в списке, отметьте подтверждение и нажмите **Delete selected model**.

### Chat Playground

В разделе **Playground** можно тестировать модель прямо в панели. В настройках генерации доступны:

- **System prompt** — системное сообщение
- **Temperature** (0.0–2.0) — креативность генерации
- **Max tokens** (1–16384) — максимальная длина ответа
- **Top-p** (0.0–1.0) — nucleus sampling
- **Repetition penalty** (1.0–2.0) — штраф за повторы

Если ответ обрезан по лимиту max_tokens, в конце сообщения появится предупреждение.

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
| `SAFETENSORS_HEALTH_TIMEOUT` | `600` секунд | Таймаут health check при старте |
| `SAFETENSORS_STOP_TIMEOUT` | `30` секунд | Таймаут graceful shutdown |
| `SAFETENSORS_AUTO_RESTART` | `0` (выключено) | Автоматический перезапуск vLLM при краше |
| `SAFETENSORS_AUTO_RESTART_MAX_RETRIES` | `3` | Максимум попыток авто-рестарта |
| `SAFETENSORS_MAX_LOG_BYTES` | `52428800` (50 MB) | Лимит размера vllm.log перед ротацией |

Старые `GGUF_*` имена читаются как fallback для API/panel/paths, но для новых deployments используйте `SAFETENSORS_*`.

# Safetensors Inference Rig

RunPod-панель для запуска Hugging Face моделей в формате Safetensors через vLLM. Один и тот же checkpoint можно активировать с вычислениями `BF16`, `FP16` или `FP32`; API совместим с OpenAI.

## Возможности

- поиск локальных моделей в `/workspace/models/safetensors/<org>/<repo>`;
- проверка Hugging Face-репозитория и скачивание полного snapshot: `.safetensors`, `config.json`, tokenizer и model code;
- альтернативные веса `.bin`, `.gguf`, ONNX и другие тяжёлые форматы при скачивании пропускаются;
- выбор `BF16`, `FP16` или `FP32` при активации;
- настройки context length, concurrent sequences, tensor parallel и доли GPU memory;
- один управляемый vLLM server с health check, graceful shutdown, сохранением активной конфигурации и rollback;
- OpenAI-compatible API с Bearer key и Gradio с Basic Auth.

> Выбор dtype задаёт вычислительный dtype vLLM. Он не переписывает исходные `.safetensors` на диске. FP32 обычно требует примерно вдвое больше памяти весов, чем BF16/FP16. BF16 требует совместимую GPU (обычно NVIDIA Ampere или новее).

## RunPod

Используйте актуальный CUDA/PyTorch Ubuntu template, network volume с mount path `/workspace` и откройте HTTP-порты `7860,8000`.

Secrets / Environment Variables:

```text
SAFETENSORS_API_KEY=<случайный длинный токен>
SAFETENSORS_PANEL_USER=<логин>
SAFETENSORS_PANEL_PASSWORD=<сложный пароль>
HF_TOKEN=<необязательно; нужен для gated/private моделей>
```

Команду запуска возьмите из `runpod_command.txt`. Первый запуск создаёт persistent venv. По умолчанию venv создаётся с `--system-site-packages`, поэтому он видит пакеты из RunPod template. Если `vllm` уже есть в template, bootstrap не будет устанавливать его заново и не должен тянуть тяжёлые `nvidia-*` wheels. После успешной установки можно задать `SAFETENSORS_SKIP_INSTALL=1`.

Если используете template без vLLM, оставьте `SAFETENSORS_INSTALL_VLLM=auto` или задайте `SAFETENSORS_INSTALL_VLLM=1`. Если используете template с готовым vLLM/PyTorch/CUDA стеком и хотите запретить установку vLLM через pip, задайте `SAFETENSORS_INSTALL_VLLM=0`; запуск остановится с ошибкой, если `vllm` не импортируется.

## Работа в панели

1. Введите `organization/repository` и нажмите **Inspect repository**.
2. Нажмите **Download snapshot**. Модель появится в локальном списке только после наличия `config.json` и хотя бы одного `.safetensors`.
3. Выберите dtype и параметры запуска, затем **Activate model**.
4. Для репозиториев с собственным Python-кодом включайте `Trust repository remote code` только если доверяете автору.

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

| Variable | Default |
| --- | --- |
| `SAFETENSORS_VOLUME_ROOT` | `/workspace` |
| `SAFETENSORS_MODELS_DIR` | `/workspace/models/safetensors` |
| `SAFETENSORS_STATE_DIR` | `/workspace/.state/safetensors-rig` |
| `SAFETENSORS_LOG_DIR` | `/workspace/logs/safetensors-rig` |
| `VLLM_PYTHON` | текущий Python interpreter |
| `SAFETENSORS_API_PORT` | `8000` |
| `SAFETENSORS_PANEL_PORT` | `7860` |
| `SAFETENSORS_HEALTH_TIMEOUT` | `600` секунд |
| `SAFETENSORS_STOP_TIMEOUT` | `30` секунд |

Старые `GGUF_*` имена читаются как fallback для API/panel/paths, но для новых deployments используйте `SAFETENSORS_*`.

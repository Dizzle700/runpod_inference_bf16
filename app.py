#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
import threading
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def _resolve_default_volume() -> Path:
    """Determine the default volume root from environment or filesystem."""
    env_volume = os.environ.get("SAFETENSORS_VOLUME_ROOT") or os.environ.get("GGUF_VOLUME_ROOT")
    if env_volume:
        return Path(env_volume).expanduser()
    if Path("/workspace").is_dir():
        return Path("/workspace")
    return PROJECT_DIR / "data"


DEFAULT_VOLUME = _resolve_default_volume()
os.environ.setdefault("HF_HOME", str(DEFAULT_VOLUME / ".hf"))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_VOLUME / ".hf" / "hub"))

import gradio as gr  # noqa: E402

from gguf_rig import ActiveModel, ChatClient, DownloadCancelled, ModelLibrary, RigConfig, VllmServerManager  # noqa: E402
from gguf_rig.system import disk_stats, gpu_stats  # noqa: E402


CSS = """
:root { --rig-amber: #e8a33d; --rig-cyan: #4fd1c5; }
.gradio-container { max-width: 1180px !important; }
.rig-hero { padding: 18px 20px; border: 1px solid var(--border-color-primary); border-radius: 12px;
  background: radial-gradient(circle at 10% 0%, rgba(232,163,61,.14), transparent 44%),
              radial-gradient(circle at 95% 0%, rgba(79,209,197,.10), transparent 40%); }
.rig-hero h1 { margin: 0 0 4px; font-size: 1.7rem; }
.rig-hero p { margin: 0; opacity: .72; }
.rig-status { min-height: 164px; }
.rig-note { opacity: .78; font-size: .92rem; }
.active-model-box { padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary); margin-bottom: 14px; font-size: 0.95rem; }
footer { display: none !important; }
"""

config = RigConfig.from_env()
config.ensure_directories()
library = ModelLibrary(config)
manager = VllmServerManager(config, library)
download_cancel_event = threading.Event()
chat_client = ChatClient(config, manager)


def _format_uptime(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def dashboard_markdown() -> str:
    status = manager.status()
    disk = disk_stats(config.models_dir)
    gpus = gpu_stats()
    state_icon = "🟢" if status["healthy"] else ("🟠" if status["running"] else "⚫")
    model = status["model"] or "none"
    if status["dtype"]:
        model += f" ({status['dtype']})"
    lines = [
        f"### {state_icon} vLLM: `{status['state']}`",
        f"- **Model:** `{model}`",
        f"- **PID / uptime:** `{status['pid'] or '—'}` / `{_format_uptime(status['uptime_seconds'])}`",
        f"- **Volume:** `{disk['free_gib']:.1f} GiB free` / `{disk['total_gib']:.1f} GiB`",
        f"- **Local API:** `{status['api_url']}`",
    ]

    # Active model parameters.
    if status["model"]:
        params = []
        if status.get("max_model_len"):
            params.append(f"ctx {status['max_model_len']}")
        if status.get("tensor_parallel_size") and status["tensor_parallel_size"] > 1:
            params.append(f"TP={status['tensor_parallel_size']}")
        if status.get("gpu_memory_utilization"):
            params.append(f"GPU util {status['gpu_memory_utilization']:.0%}")
        if status.get("enforce_eager"):
            params.append("eager")
        if status.get("enable_chunked_prefill"):
            params.append("chunked-prefill")
        if status.get("max_audio_per_prompt"):
            params.append(f"audio ≤{status['max_audio_per_prompt']}/request")
        if params:
            lines.append(f"- **Params:** {' · '.join(params)}")

    # Auto-restart indicator.
    if status.get("auto_restart"):
        lines.append("- **Auto-restart:** ✅ enabled")

    # GPU info.
    if gpus:
        for gpu in gpus:
            lines.append(
                f"- **GPU {gpu['index']}:** {gpu['name']} · "
                f"{gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB · "
                f"{gpu['utilization']}% · {gpu['temperature']}°C"
            )
    else:
        lines.append("- **GPU:** `nvidia-smi unavailable`")

    # vLLM metrics (only when healthy).
    if status["healthy"]:
        m = manager.metrics()
        metric_parts = []
        if m.get("active_requests") is not None:
            metric_parts.append(f"active {int(m['active_requests'])}")
        if m.get("pending_requests") is not None:
            metric_parts.append(f"pending {int(m['pending_requests'])}")
        if m.get("kv_cache_usage_percent") is not None:
            metric_parts.append(f"KV cache {m['kv_cache_usage_percent']:.1%}")
        if m.get("avg_generation_throughput") is not None:
            metric_parts.append(f"{m['avg_generation_throughput']:.1f} tok/s")
        if metric_parts:
            lines.append(f"- **vLLM:** {' · '.join(metric_parts)}")

    # API statistics.
    stats = manager.api_stats()
    if stats["total_requests"] > 0:
        lines.append(
            f"- **API stats:** {stats['total_requests']} reqs · "
            f"{stats['total_tokens']} tokens · "
            f"avg {stats['avg_latency_s']:.2f}s · "
            f"{stats['errors']} errors"
        )

    # Model count.
    model_count = len(library.scan())
    lines.append(f"- **Models on volume:** {model_count}")

    return "\n".join(lines)


def _model_choices() -> list[tuple[str, str]]:
    return [
        (
            f"{record.id} · {record.dtype_label} · "
            f"{record.shard_count} shard(s) · {record.size_gib:.2f} GiB · "
            f"{record.revision or 'local'}@{record.commit_sha[:8] or 'unknown'}",
            record.id,
        )
        for record in library.scan()
    ]


def refresh_library(selected: str | None = None):
    choices = _model_choices()
    values = {value for _, value in choices}
    value = selected if selected in values else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value), dashboard_markdown()


def library_status_markdown() -> str:
    statuses = library.statuses()
    if not statuses:
        return "_No local or in-progress models._"
    lines = ["| Model | Revision | Commit | Status |", "| --- | --- | --- | --- |"]
    icons = {"ready": "✅", "downloading": "⬇️", "incomplete": "⚠️", "broken": "❌"}
    for item in statuses:
        lines.append(
            f"| `{item['model']}` | `{item['revision']}` | `{item['commit']}` | "
            f"{icons.get(item['status'], '⚠️')} {item['status']} |"
        )
    return "\n".join(lines)


def event_history_markdown() -> str:
    events = manager.recent_events(50)
    if not events:
        return "_No lifecycle events recorded yet._"
    lines = ["| Time (UTC) | Event | Model | Detail |", "| --- | --- | --- | --- |"]
    for event in reversed(events):
        detail = event.get("error") or (
            f"ctx={event['max_model_len']}" if event.get("max_model_len") else ""
        )
        lines.append(
            f"| `{event.get('timestamp', '')}` | `{event.get('event', '')}` | "
            f"`{event.get('model') or '—'}` | {html.escape(str(detail))} |"
        )
    return "\n".join(lines)


def inspect_remote(repo_id: str, revision: str):
    try:
        remote = library.inspect_remote(repo_id, token=config.hf_token or None, revision=revision)
        return (
            f"✅ **{remote.repo_id}** · {remote.shard_count} Safetensors file(s) · "
            f"{remote.size_label} · metadata dtype: `{remote.config_dtype}` · "
            f"revision: `{remote.revision}` · commit: `{remote.commit_sha}` · "
            f"HF token: **{'configured' if config.hf_token else 'not configured'}**"
        )
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}"


def download_remote(repo_id: str, revision: str, progress=gr.Progress()):
    download_cancel_event.clear()
    try:
        remote = library.inspect_remote(
            repo_id, token=config.hf_token or None, revision=revision
        )

        def report(value: float, description: str) -> None:
            progress(value, desc=description)

        path = library.download_snapshot(
            remote.repo_id,
            token=config.hf_token or None,
            revision=remote.revision,
            commit_sha=remote.commit_sha,
            expected_size=remote.size_bytes,
            progress=report,
            cancelled=download_cancel_event.is_set,
        )
        choices = _model_choices()
        model_id = path.relative_to(config.models_dir.resolve()).as_posix()
        return f"✅ Downloaded complete model snapshot to `{path}`", gr.update(choices=choices, value=model_id)
    except DownloadCancelled as exc:
        return f"⚠️ {html.escape(str(exc))}", gr.update()
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", gr.update()


def cancel_download():
    download_cancel_event.set()
    return "⚠️ Cancellation requested; the current file transfer will stop at its next progress update."


def delete_model(model_id: str | None, confirm_delete: bool):
    if not model_id:
        return "❌ Select a model to delete.", gr.update(), dashboard_markdown()
    if not confirm_delete:
        return "⚠️ Check the deletion confirmation box.", gr.update(), dashboard_markdown()
    # Prevent deleting the currently active model.
    status = manager.status()
    if status["running"] and status["model"] == model_id:
        return "❌ Cannot delete the currently active model. Stop or switch first.", gr.update(), dashboard_markdown()
    try:
        result = library.delete(model_id)
        choices = _model_choices()
        new_value = choices[0][1] if choices else None
        return f"✅ {result}", gr.update(choices=choices, value=new_value), dashboard_markdown()
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", gr.update(), dashboard_markdown()


def activate_model(
    model_id: str | None,
    dtype: str,
    max_model_len: float,
    max_num_seqs: float,
    tensor_parallel_size: float,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    chat_template: str,
    enforce_eager: bool,
    enable_chunked_prefill: bool,
    auto_adjust_context: bool,
    max_audio_per_prompt: float,
    confirm_switch: bool,
):
    if not model_id:
        return "❌ Select a downloaded model.", dashboard_markdown()
    current = manager.status()
    if current["running"] and not confirm_switch:
        return "⚠️ Check the switch confirmation box; active requests may be interrupted.", dashboard_markdown()
    active = ActiveModel(
        model_id=model_id,
        dtype=dtype,
        max_model_len=int(max_model_len),
        max_num_seqs=int(max_num_seqs),
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        trust_remote_code=bool(trust_remote_code),
        chat_template=(chat_template or "").strip(),
        enforce_eager=bool(enforce_eager),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        auto_adjust_context=bool(auto_adjust_context),
        max_audio_per_prompt=int(max_audio_per_prompt),
    )
    try:
        return f"✅ {manager.switch(active)}", dashboard_markdown()
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", dashboard_markdown()


def restart_server():
    try:
        return f"✅ {manager.restart()}", dashboard_markdown()
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", dashboard_markdown()


def stop_server():
    try:
        return f"✅ {manager.stop()}", dashboard_markdown()
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", dashboard_markdown()


def active_model_info() -> str:
    status = manager.status()
    if status["healthy"] and status["model"]:
        model_name = status["model"]
        if status.get("dtype"):
            model_name += f" ({status['dtype']})"
        return f"🟢 **Active Model:** `{model_name}`"
    elif status["healthy"]:
        return "🟢 **Active Model:** *(starting/ready)*"
    elif status["running"]:
        return "🟠 **Active Model:** *(initializing server...)*"
    else:
        return "🔴 **Active Model:** `None` *(server stopped)*"


def build_app() -> gr.Blocks:
    choices = _model_choices()
    initial_model = choices[0][1] if choices else None
    with gr.Blocks(title="Safetensors Inference Rig", css=CSS, theme=gr.themes.Base()) as demo:
        gr.HTML(
            "<div class='rig-hero'>"
            "<h1>Safetensors Inference Rig</h1>"
            "<p>BF16 · FP16 · FP32 · vLLM · OpenAI-compatible API</p>"
            "</div>"
        )
        with gr.Tabs():
            with gr.Tab("Dashboard"):
                dashboard = gr.Markdown(dashboard_markdown(), elem_classes="rig-status")
                with gr.Row():
                    refresh_dashboard = gr.Button("Refresh", variant="secondary")
                    restart_button = gr.Button("Restart server")
                    stop_button = gr.Button("Stop server", variant="stop")
                dashboard_message = gr.Markdown()

            with gr.Tab("Model Library"):
                with gr.Row():
                    model_select = gr.Dropdown(
                        label="Safetensors models on persistent volume",
                        choices=choices,
                        value=initial_model,
                        filterable=True,
                        scale=3,
                    )
                    refresh_models = gr.Button("Rescan volume", scale=1)
                with gr.Accordion("Run configuration", open=True):
                    dtype = gr.Dropdown(
                        label="Compute dtype",
                        choices=[
                            ("BF16 (recommended on Ampere+)", "bfloat16"),
                            ("FP16", "float16"),
                            ("FP32 (very high VRAM use)", "float32"),
                        ],
                        value="bfloat16",
                    )
                    with gr.Row():
                        max_model_len = gr.Number(label="Max model length", value=8192, precision=0, minimum=512)
                        max_num_seqs = gr.Number(label="Max concurrent sequences", value=64, precision=0, minimum=1)
                        tensor_parallel_size = gr.Number(label="Tensor parallel GPUs", value=1, precision=0, minimum=1)
                        max_audio_per_prompt = gr.Number(
                            label="Max audio clips per request", value=1, precision=0, minimum=1, maximum=8
                        )
                        gpu_memory_utilization = gr.Slider(
                            label="GPU memory utilization", value=0.90, minimum=0.05, maximum=1.0, step=0.01
                        )
                    with gr.Row():
                        trust_remote_code = gr.Checkbox(
                            label="Trust repository remote code (enable only for repositories you trust)"
                        )
                        enforce_eager = gr.Checkbox(
                            label="Enforce eager mode (skip CUDA graphs, saves VRAM & speeds up startup)"
                        )
                        enable_chunked_prefill = gr.Checkbox(
                            label="Enable chunked prefill (helps prevent OOM on long contexts)"
                        )
                        auto_adjust_context = gr.Checkbox(
                            label="Auto-reduce context after KV-cache/OOM startup failure",
                            value=True,
                        )
                    chat_template = gr.Textbox(label="Chat-template override (normally empty)", lines=3)
                confirm_switch = gr.Checkbox(label="I understand that switching can interrupt in-flight requests")
                activate_button = gr.Button("Activate model", variant="primary")
                activation_result = gr.Markdown()

                gr.Markdown("---")

                gr.Markdown("### Download from Hugging Face")
                gr.Markdown(
                    "Downloads the complete model snapshot: Safetensors weights, config, tokenizer and model code. "
                    "Alternative `.bin`, GGUF and ONNX weights are skipped.",
                    elem_classes="rig-note",
                )
                repo_id = gr.Textbox(
                    label="Repository",
                    placeholder="organization/repository or https://huggingface.co/organization/repository",
                )
                revision = gr.Textbox(
                    label="Revision",
                    value="main",
                    placeholder="branch, tag, or commit SHA",
                )
                with gr.Row():
                    inspect_button = gr.Button("Inspect repository")
                    download_button = gr.Button("Download snapshot", variant="primary")
                    cancel_download_button = gr.Button("Cancel download", variant="stop")
                remote_summary = gr.Markdown()
                download_result = gr.Markdown()
                gr.Markdown("### Model status")
                library_status = gr.Markdown(library_status_markdown())

                gr.Markdown("---")

                gr.Markdown("### Delete Model")
                gr.Markdown(
                    "Permanently removes a downloaded model from the persistent volume.",
                    elem_classes="rig-note",
                )
                confirm_delete = gr.Checkbox(
                    label="I understand this permanently deletes the model files and cannot be undone"
                )
                delete_button = gr.Button("Delete selected model", variant="stop")
                delete_result = gr.Markdown()

            with gr.Tab("Playground"):
                active_model_box = gr.Markdown(active_model_info(), elem_classes="active-model-box")
                gr.Markdown("### Chat Playground")
                audio_file = gr.Audio(
                    label="Audio input (optional)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                gr.Markdown(
                    "For Audio Flamingo Next, upload one mono 16 kHz clip and ask a direct "
                    "question (ASR, captioning, translation, timestamps, etc.). "
                    f"Uploads are limited to {config.max_audio_upload_bytes / 1024**2:.0f} MiB.",
                    elem_classes="rig-note",
                )
                with gr.Accordion("Generation settings", open=False):
                    system_prompt = gr.Textbox(
                        label="System prompt",
                        placeholder="You are a helpful assistant...",
                        lines=2,
                    )
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature", value=0.7, minimum=0.0, maximum=2.0, step=0.05
                        )
                        max_tokens = gr.Slider(
                            label="Max tokens", value=2048, minimum=1, maximum=16384, step=1
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p", value=1.0, minimum=0.0, maximum=1.0, step=0.01
                        )
                        min_p = gr.Slider(
                            label="Min-p", value=0.0, minimum=0.0, maximum=1.0, step=0.01
                        )
                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition penalty", value=1.0, minimum=1.0, maximum=2.0, step=0.01
                        )
                        top_k = gr.Slider(
                            label="Top-k (-1 to disable)", value=-1, minimum=-1, maximum=100, step=1
                        )
                    with gr.Row():
                        presence_penalty = gr.Slider(
                            label="Presence penalty", value=0.0, minimum=-2.0, maximum=2.0, step=0.05
                        )
                        frequency_penalty = gr.Slider(
                            label="Frequency penalty", value=0.0, minimum=-2.0, maximum=2.0, step=0.05
                        )
                gr.ChatInterface(
                    fn=chat_client.stream,
                    type="messages",
                    additional_inputs=[
                        system_prompt, temperature, max_tokens, top_p,
                        repetition_penalty, top_k, presence_penalty, frequency_penalty, min_p,
                        audio_file,
                    ],
                    examples=[
                        ["Explain quantum computing in simple terms."],
                        ["Write a Python function to check if a number is prime."],
                        ["Draft a professional response to a customer complaining about a delayed delivery."],
                        ["Act as a creative naming assistant and suggest 10 names for a new AI coding tool."],
                    ],
                )

            with gr.Tab("Console"):
                console = gr.Textbox(label="vLLM output", value=manager.logs(), lines=28, interactive=False)
                console_refresh = gr.Button("Refresh log")
                gr.Markdown("### Launch/error history")
                event_history = gr.Markdown(event_history_markdown())

            with gr.Tab("Settings"):
                gr.Markdown(f"""
### Runtime

- Model volume: `{config.models_dir}`
- State: `{config.state_dir}`
- vLLM Python: `{config.python_executable}`
- API listener: `{config.api_host}:{config.api_port}`
- Panel port fallback: **{"enabled (local development)" if config.panel_find_free_port else "disabled (strict bind)"}**
- API key: **{"configured" if config.api_key else "missing"}**
- Panel authentication: **{"configured" if config.panel_user and config.panel_password else "missing"}**
- Hugging Face token: **{"configured" if config.hf_token else "not configured"}**

### Process Management

- Auto-restart on crash: **{"enabled" if config.auto_restart else "disabled"}** (`SAFETENSORS_AUTO_RESTART`)
- Auto-restart max retries: **{config.auto_restart_max_retries}** (`SAFETENSORS_AUTO_RESTART_MAX_RETRIES`)
- Max log file size: **{config.max_log_bytes / 1024 / 1024:.0f} MB** (`SAFETENSORS_MAX_LOG_BYTES`)
- Health check timeout: **{config.health_timeout}s** (`SAFETENSORS_HEALTH_TIMEOUT`)
- Stop timeout: **{config.stop_timeout}s** (`SAFETENSORS_STOP_TIMEOUT`)

Secrets are environment-only. Change them in RunPod Secrets and restart the pod.
                """)

        # -- Event wiring --------------------------------------------------

        refresh_dashboard.click(dashboard_markdown, outputs=dashboard)
        restart_button.click(
            restart_server,
            outputs=[dashboard_message, dashboard],
            concurrency_limit=1,
            concurrency_id="model_operations",
        )
        stop_button.click(
            stop_server,
            outputs=[dashboard_message, dashboard],
            concurrency_limit=1,
            concurrency_id="model_operations",
        )
        refresh_models.click(refresh_library, inputs=model_select, outputs=[model_select, dashboard])
        activate_button.click(
            activate_model,
            inputs=[
                model_select, dtype, max_model_len, max_num_seqs,
                tensor_parallel_size, gpu_memory_utilization, trust_remote_code,
                chat_template, enforce_eager, enable_chunked_prefill,
                auto_adjust_context, max_audio_per_prompt, confirm_switch,
            ],
            outputs=[activation_result, dashboard],
            concurrency_limit=1,
            concurrency_id="model_operations",
        )
        inspect_button.click(inspect_remote, inputs=[repo_id, revision], outputs=remote_summary)
        download_button.click(
            download_remote,
            inputs=[repo_id, revision],
            outputs=[download_result, model_select],
            concurrency_limit=1,
            concurrency_id="model_operations",
        )
        cancel_download_button.click(
            cancel_download,
            outputs=download_result,
            queue=False,
        )
        delete_button.click(
            delete_model,
            inputs=[model_select, confirm_delete],
            outputs=[delete_result, model_select, dashboard],
            concurrency_limit=1,
            concurrency_id="model_operations",
        )
        console_refresh.click(lambda: manager.logs(), outputs=console)
        timer = gr.Timer(5)
        timer.tick(dashboard_markdown, outputs=dashboard)
        timer.tick(active_model_info, outputs=active_model_box)
        timer.tick(lambda: manager.logs(), outputs=console)
        timer.tick(library_status_markdown, outputs=library_status)
        timer.tick(event_history_markdown, outputs=event_history)
    return demo


def _restore_in_background() -> None:
    try:
        manager.restore()
    except Exception as exc:
        manager.append_log(f"Automatic restore failed: {exc}")


def find_free_port(host: str, start_port: int) -> int:
    import socket
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1
    return start_port


def main() -> None:
    from fastapi import FastAPI
    import uvicorn

    config.validate_security()
    demo = build_app()
    threading.Thread(target=_restore_in_background, daemon=True).start()
    auth = (config.panel_user, config.panel_password) if config.panel_user and config.panel_password else None

    port = find_free_port(config.panel_host, config.panel_port)
    if port != config.panel_port and not config.panel_find_free_port:
        raise OSError(
            f"Configured panel port {config.panel_port} is already in use; "
            "strict binding is enabled"
        )
    if port != config.panel_port:
        print(f"⚠️ Port {config.panel_port} was busy. Switched to {port}.")

    web_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @web_app.get("/healthz")
    def healthz():
        status = manager.health_snapshot()
        return {
            "status": "ok",
            "panel": "ready",
            "vllm": status["state"],
            "model": status["model"],
        }

    demo.queue(default_concurrency_limit=4)
    web_app = gr.mount_gradio_app(
        web_app,
        demo,
        path="/",
        auth=auth,
        server_name=config.panel_host,
        server_port=port,
    )
    uvicorn.run(web_app, host=config.panel_host, port=port, log_level="info")


if __name__ == "__main__":
    main()

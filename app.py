#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
import threading
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VOLUME = Path(os.environ.get("SAFETENSORS_VOLUME_ROOT", os.environ.get("GGUF_VOLUME_ROOT", "/workspace" if Path("/workspace").is_dir() else str(PROJECT_DIR / "data")))).expanduser()
os.environ.setdefault("HF_HOME", str(DEFAULT_VOLUME / ".hf"))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_VOLUME / ".hf" / "hub"))

import gradio as gr  # noqa: E402

from gguf_rig import ActiveModel, ModelLibrary, RigConfig, VllmServerManager  # noqa: E402
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
footer { display: none !important; }
"""

config = RigConfig.from_env()
config.ensure_directories()
library = ModelLibrary(config)
manager = VllmServerManager(config, library)


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
    if gpus:
        for gpu in gpus:
            lines.append(f"- **GPU {gpu['index']}:** {gpu['name']} · {gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB · {gpu['utilization']}% · {gpu['temperature']}°C")
    else:
        lines.append("- **GPU:** `nvidia-smi unavailable`")
    return "\n".join(lines)


def _model_choices() -> list[tuple[str, str]]:
    return [(f"{record.id} · {record.dtype_label} · {record.shard_count} shard(s) · {record.size_gib:.2f} GiB", record.id) for record in library.scan()]


def refresh_library(selected: str | None = None):
    choices = _model_choices()
    values = {value for _, value in choices}
    value = selected if selected in values else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value), dashboard_markdown()


def inspect_remote(repo_id: str):
    try:
        remote = library.inspect_remote(repo_id, token=config.hf_token or None)
        return (
            f"✅ **{remote.repo_id}** · {remote.shard_count} Safetensors file(s) · "
            f"{remote.size_label} · metadata dtype: `{remote.config_dtype}` · "
            f"HF token: **{'configured' if config.hf_token else 'not configured'}**"
        )
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}"


def download_remote(repo_id: str, progress=gr.Progress()):
    try:
        remote = library.inspect_remote(repo_id, token=config.hf_token or None)

        def report(value: float, description: str) -> None:
            progress(value, desc=description)

        path = library.download_snapshot(remote.repo_id, token=config.hf_token or None, expected_size=remote.size_bytes, progress=report)
        choices = _model_choices()
        model_id = path.relative_to(config.models_dir.resolve()).as_posix()
        return f"✅ Downloaded complete model snapshot to `{path}`", gr.update(choices=choices, value=model_id)
    except Exception as exc:
        return f"❌ {html.escape(str(exc))}", gr.update()


def activate_model(model_id: str | None, dtype: str, max_model_len: float, max_num_seqs: float, tensor_parallel_size: float, gpu_memory_utilization: float, trust_remote_code: bool, chat_template: str, enforce_eager: bool, enable_chunked_prefill: bool, confirm_switch: bool):
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


def chat_stream(message: str, history: list[dict] | list[Any]):
    status = manager.status()
    if not status["healthy"]:
        yield "❌ Server is not ready or stopped. Please activate a model first."
        return

    messages = []
    for h in history:
        if isinstance(h, dict):
            messages.append({"role": h["role"], "content": h["content"]})
        else:
            messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": message})

    url = f"{config.local_api_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    data = {
        "model": "current",
        "messages": messages,
        "stream": True,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            buffer = ""
            accumulated = ""
            for chunk in response:
                decoded = chunk.decode("utf-8")
                buffer += decoded
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            payload = json.loads(data_str)
                            choice = payload.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated += content
                                yield accumulated
                        except Exception:
                            pass
    except urllib.error.HTTPError as e:
        yield f"❌ HTTP Error {e.code}: {e.reason}"
    except Exception as e:
        yield f"❌ Error: {e}"


def build_app() -> gr.Blocks:
    choices = _model_choices()
    initial_model = choices[0][1] if choices else None
    with gr.Blocks(title="Safetensors Inference Rig", css=CSS, theme=gr.themes.Base()) as demo:
        gr.HTML("<div class='rig-hero'><h1>Safetensors Inference Rig</h1><p>BF16 · FP16 · FP32 · vLLM · OpenAI-compatible API</p></div>")
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
                    model_select = gr.Dropdown(label="Safetensors models on persistent volume", choices=choices, value=initial_model, filterable=True, scale=3)
                    refresh_models = gr.Button("Rescan volume", scale=1)
                with gr.Accordion("Run configuration", open=True):
                    dtype = gr.Dropdown(
                        label="Compute dtype",
                        choices=[("BF16 (recommended on Ampere+)", "bfloat16"), ("FP16", "float16"), ("FP32 (very high VRAM use)", "float32")],
                        value="bfloat16",
                    )
                    with gr.Row():
                        max_model_len = gr.Number(label="Max model length", value=8192, precision=0, minimum=512)
                        max_num_seqs = gr.Number(label="Max concurrent sequences", value=64, precision=0, minimum=1)
                        tensor_parallel_size = gr.Number(label="Tensor parallel GPUs", value=1, precision=0, minimum=1)
                        gpu_memory_utilization = gr.Slider(label="GPU memory utilization", value=0.90, minimum=0.05, maximum=1.0, step=0.01)
                    with gr.Row():
                        trust_remote_code = gr.Checkbox(label="Trust repository remote code (enable only for repositories you trust)")
                        enforce_eager = gr.Checkbox(label="Enforce eager mode (skip CUDA graphs, saves VRAM & speeds up startup)")
                        enable_chunked_prefill = gr.Checkbox(label="Enable chunked prefill (helps prevent OOM on long contexts)")
                    chat_template = gr.Textbox(label="Chat-template override (normally empty)", lines=3)
                confirm_switch = gr.Checkbox(label="I understand that switching can interrupt in-flight requests")
                activate_button = gr.Button("Activate model", variant="primary")
                activation_result = gr.Markdown()

                gr.Markdown("### Download from Hugging Face")
                gr.Markdown("Downloads the complete model snapshot: Safetensors weights, config, tokenizer and model code. Alternative `.bin`, GGUF and ONNX weights are skipped.", elem_classes="rig-note")
                repo_id = gr.Textbox(label="Repository", placeholder="organization/repository or https://huggingface.co/organization/repository")
                with gr.Row():
                    inspect_button = gr.Button("Inspect repository")
                    download_button = gr.Button("Download snapshot", variant="primary")
                remote_summary = gr.Markdown()
                download_result = gr.Markdown()

            with gr.Tab("Playground"):
                gr.Markdown("### Chat Playground")
                gr.ChatInterface(
                    fn=chat_stream,
                    type="messages",
                )

            with gr.Tab("Console"):
                console = gr.Textbox(label="vLLM output", value=manager.logs(), lines=28, interactive=False)
                console_refresh = gr.Button("Refresh log")

            with gr.Tab("Settings"):
                gr.Markdown(f"""
### Runtime

- Model volume: `{config.models_dir}`
- State: `{config.state_dir}`
- vLLM Python: `{config.python_executable}`
- API listener: `{config.api_host}:{config.api_port}`
- API key: **{"configured" if config.api_key else "missing"}**
- Panel authentication: **{"configured" if config.panel_user and config.panel_password else "missing"}**
- Hugging Face token: **{"configured" if config.hf_token else "not configured"}**

Secrets are environment-only. Change them in RunPod Secrets and restart the pod.
                """)

        refresh_dashboard.click(dashboard_markdown, outputs=dashboard)
        restart_button.click(restart_server, outputs=[dashboard_message, dashboard])
        stop_button.click(stop_server, outputs=[dashboard_message, dashboard])
        refresh_models.click(refresh_library, inputs=model_select, outputs=[model_select, dashboard])
        activate_button.click(activate_model, inputs=[model_select, dtype, max_model_len, max_num_seqs, tensor_parallel_size, gpu_memory_utilization, trust_remote_code, chat_template, enforce_eager, enable_chunked_prefill, confirm_switch], outputs=[activation_result, dashboard], concurrency_limit=1)
        inspect_button.click(inspect_remote, inputs=repo_id, outputs=remote_summary)
        download_button.click(download_remote, inputs=repo_id, outputs=[download_result, model_select], concurrency_limit=1)
        console_refresh.click(lambda: manager.logs(), outputs=console)
        timer = gr.Timer(5)
        timer.tick(dashboard_markdown, outputs=dashboard)
        timer.tick(lambda: manager.logs(), outputs=console)
    return demo


def _restore_in_background() -> None:
    try:
        manager.restore()
    except Exception as exc:
        manager._append_log(f"Automatic restore failed: {exc}")


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
    config.validate_security()
    demo = build_app()
    threading.Thread(target=_restore_in_background, daemon=True).start()
    auth = (config.panel_user, config.panel_password) if config.panel_user and config.panel_password else None
    port = find_free_port(config.panel_host, config.panel_port)
    if port != config.panel_port:
        print(f"⚠️ Port {config.panel_port} was busy. Switched to {port}.")
    demo.queue(default_concurrency_limit=4).launch(server_name=config.panel_host, server_port=port, auth=auth, show_error=True)


if __name__ == "__main__":
    main()

import multiprocessing
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Union


def _run_download_task(args: List[str], event_queue) -> None:
    try:
        from app.core.download_script import run_download_task

        run_download_task(args, event_queue)
    except Exception as exc:
        try:
            event_queue.put({"type": "error", "message": str(exc)})
        except Exception:
            pass
        try:
            event_queue.put({"type": "done"})
        except Exception:
            pass


class DownloadService:
    def __init__(self, script_path: str, cache_dir: str, models_dir: str) -> None:
        self._script_path = script_path
        self._cache_dir = cache_dir
        self._models_dir = models_dir

        self._lock = threading.Lock()
        self._process: Optional[Union[subprocess.Popen, multiprocessing.Process]] = None
        self._ipc_queue: Optional[object] = None
        self._queue: Optional[queue.Queue] = None
        self._reader: Optional[threading.Thread] = None
        self._running = False
        self._is_subprocess: bool = False
        self._status: Dict[str, object] = {
            "running": False,
            "repo_id": "",
            "percent": 0,
            "file": "",
            "message": "",
            "error": "",
            "path": "",
            "started_at": None,
            "updated_at": None,
        }

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._status)

    def _update_status(self, **kwargs) -> None:
        now = time.time()
        with self._lock:
            self._status.update(kwargs)
            self._status["updated_at"] = now

    def start(self, repo_id: str) -> queue.Queue:
        with self._lock:
            if self._running:
                raise RuntimeError("Download already running")
            model_name = repo_id.split("/")[-1].strip()
            if model_name:
                candidates = [model_name]
                replaced = model_name.replace(".", "___")
                if replaced != model_name:
                    candidates.append(replaced)
                models_root = Path(self._models_dir)
                for name in candidates:
                    if (models_root / name).exists():
                        raise RuntimeError(f"模型已存在: {name}")

            # 检测是否在 PyInstaller 打包环境中
            is_frozen = getattr(sys, "frozen", False)
            self._queue = queue.Queue()

            if is_frozen:
                # 打包环境：使用 subprocess 启动 EXE 配合 --download-script 参数
                self._is_subprocess = True
                args = [
                    sys.executable,
                    "--download-script",
                    repo_id,
                    self._cache_dir,
                    self._models_dir,
                ]
                self._process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                )
                self._reader = threading.Thread(
                    target=self._read_subprocess_output, daemon=True
                )
            else:
                # 开发环境：使用 multiprocessing.Process 直接调用函数
                self._is_subprocess = False
                ctx = multiprocessing.get_context("spawn")
                self._ipc_queue = ctx.Queue()
                self._process = ctx.Process(
                    target=_run_download_task,
                    args=(
                        [repo_id, self._cache_dir, self._models_dir],
                        self._ipc_queue,
                    ),
                    daemon=True,
                )
                self._process.start()
                self._reader = threading.Thread(target=self._read_loop, daemon=True)

            self._running = True
            self._reader.start()
            self._status = {
                "running": True,
                "repo_id": repo_id,
                "percent": 0,
                "file": "",
                "message": "",
                "error": "",
                "path": "",
                "started_at": time.time(),
                "updated_at": time.time(),
            }

            return self._queue

    def stop(self) -> None:
        with self._lock:
            process = self._process
            is_subprocess = self._is_subprocess
            if not process:
                return
            try:
                if is_subprocess:
                    # subprocess.Popen
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
                else:
                    # multiprocessing.Process
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1)
            finally:
                self._running = False
                self._status["running"] = False
                self._status["message"] = "cancelled"
                self._status["updated_at"] = time.time()


        # 统一向队列推送取消/完成事件，确保 UI 能收到通知
        if self._queue is not None:
            try:
                self._queue.put({"type": "cancelled"})
                self._queue.put({"type": "done"})
            except Exception:
                pass

    def _handle_event(self, item: Dict[str, object]) -> None:
        event_type = item.get("type")
        if event_type == "progress":
            percent = int(item.get("percent") or 0)
            file_name = item.get("file") or ""
            self._update_status(percent=percent, file=file_name, message="")
            return
        if event_type == "finished":
            path = item.get("path") or ""
            self._update_status(path=path)
            return
        if event_type == "error":
            message = item.get("message") or ""
            self._update_status(error=message, message="")
            return
        if event_type == "cancelled":
            self._update_status(message="cancelled")
            return
        if event_type == "log":
            message = item.get("message") or ""
            if message:
                self._update_status(message=message)

    def _read_loop(self) -> None:
        assert self._ipc_queue is not None
        assert self._queue is not None

        done = False
        while True:
            try:
                item = self._ipc_queue.get(timeout=0.2)
            except queue.Empty:
                if self._process is not None and not self._process.is_alive():
                    break
                continue

            if not isinstance(item, dict):
                continue
            self._handle_event(item)
            self._queue.put(item)
            if item.get("type") == "done":
                done = True
                break

        if not done:
            exit_code = None
            if self._process is not None:
                self._process.join(timeout=0)
                exit_code = self._process.exitcode
            if exit_code not in (0, None):
                error = f"Download exited with code {exit_code}"
                self._update_status(error=error, message="")
                self._queue.put({"type": "error", "message": error})
            self._queue.put({"type": "done"})

        with self._lock:
            self._running = False
            self._status["running"] = False
            self._status["updated_at"] = time.time()

    def _read_subprocess_output(self) -> None:
        """读取打包环境下子进程的 stdout 输出（@PROGRESS@file@percent 格式）"""
        assert self._process is not None
        assert self._queue is not None

        done = False
        while True:
            try:
                line = self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                # 解析子进程输出的事件格式
                if line.startswith("@PROGRESS@"):
                    payload = line[len("@PROGRESS@"):]
                    file_name = payload
                    percent = 0
                    if "@" in payload:
                        file_name, percent_str = payload.rsplit("@", 1)
                        try:
                            percent = int(percent_str)
                        except ValueError:
                            percent = 0
                    self._update_status(percent=percent, file=file_name, message="")
                    self._queue.put({"type": "progress", "file": file_name, "percent": percent})
                elif line.startswith("@LOG@"):
                    message = line.split("@", 2)[-1]
                    self._update_status(message=message)
                    self._queue.put({"type": "log", "message": message})
                elif line.startswith("@FINISHED@"):
                    path = line.split("@", 2)[-1]
                    self._update_status(path=path)
                    self._queue.put({"type": "finished", "path": path})
                    self._queue.put({"type": "done"})
                    done = True
                    break
                elif line.startswith("@ERROR@"):
                    message = line.split("@", 2)[-1]
                    self._update_status(error=message, message="")
                    self._queue.put({"type": "error", "message": message})
                    self._queue.put({"type": "done"})
                    done = True
                    break
            except Exception:
                break

        # 等待子进程结束
        try:
            exit_code = self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            exit_code = -1

        if exit_code not in (0, None):
            error = f"Download exited with code {exit_code}"
            self._update_status(error=error, message="")
            self._queue.put({"type": "error", "message": error})
            self._queue.put({"type": "done"})

        # 向队列推送取消/完成事件，确保 UI 能收到通知

        with self._lock:
            self._running = False
            self._status["running"] = False
            self._status["updated_at"] = time.time()

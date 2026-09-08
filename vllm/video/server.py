# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-partition, serial H3 video job service."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from vllm.model_executor.models.minimax_h3.config import (
    DEFAULT_PROMPT,
    H3Config,
    H3Request,
    H3SamplingParams,
)


class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = DEFAULT_PROMPT
    width: int = 1344
    height: int = 768
    num_frames: int = 243
    duration: float | None = None
    seed: int = 42
    num_inference_steps: int = 50
    image: list[str] = Field(default_factory=list)
    video: list[str] = Field(default_factory=list)
    audio: list[str] = Field(default_factory=list)
    keyframe_indices: list[int] | None = None

    def request(self):
        extra: dict[str, Any] = {}
        if self.duration is not None:
            extra["duration_seconds"] = self.duration
        if self.keyframe_indices is not None:
            extra["frame_indices"] = self.keyframe_indices
        return H3Request(
            prompt=self.prompt,
            sampling=H3SamplingParams(
                width=self.width,
                height=self.height,
                num_frames=self.num_frames,
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
                extra_args=extra,
            ),
            media={
                key: getattr(self, key)
                for key in ("image", "video", "audio")
                if getattr(self, key)
            },
        )


def create_app(config: H3Config, output_dir: str | Path, *, engine_factory=None):
    from .engine import H3Engine

    factory = engine_factory or H3Engine
    root = Path(output_dir).absolute()
    jobs: dict[str, dict[str, Any]] = {}
    queue: asyncio.Queue[tuple[str, H3Request]] = asyncio.Queue(maxsize=32)
    state: dict[str, Any] = {
        "engine": None,
        "ready": False,
        "completed": 0,
        "failed": 0,
    }

    async def process():
        while True:
            identity, request = await queue.get()
            job = jobs[identity]
            job.update(status="in_progress", started_at=time.time())
            try:
                result = await asyncio.to_thread(
                    state["engine"].generate, request, root / identity
                )
                job.update(
                    status="completed",
                    completed_at=time.time(),
                    result=result,
                    content_url=f"/v1/videos/{identity}/content",
                )
                state["completed"] += 1
            except Exception as exc:
                job.update(status="failed", error=str(exc), completed_at=time.time())
                state["failed"] += 1
                state["ready"] = not state["engine"]._closed
            finally:
                queue.task_done()

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        state["engine"] = await asyncio.to_thread(factory, config)
        state["ready"] = True
        task = asyncio.create_task(process())
        try:
            yield
        finally:
            state["ready"] = False
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await asyncio.to_thread(state["engine"].close)

    app = FastAPI(title="1Cat H3 video API", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not state["ready"]:
            raise HTTPException(503, "video engine unavailable")
        return {"status": "ok", "partition": config.partition}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return (
            f"onecat_video_ready {int(state['ready'])}\n"
            f"onecat_video_queue_depth {queue.qsize()}\n"
            f"onecat_video_completed_total {state['completed']}\n"
            f"onecat_video_failed_total {state['failed']}\n"
        )

    @app.post("/v1/videos", status_code=202)
    async def create_video(body: VideoRequest):
        if not state["ready"]:
            raise HTTPException(503, "video engine unavailable")
        try:
            request = body.request()
            from vllm.model_executor.models.minimax_h3.validation import (
                validate_request,
            )

            await asyncio.to_thread(validate_request, config, request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if queue.full():
            raise HTTPException(429, "video queue is full")
        identity = "video_" + uuid.uuid4().hex
        jobs[identity] = {
            "id": identity,
            "status": "queued",
            "created_at": time.time(),
            "partition": config.partition,
            "request": asdict(request),
        }
        queue.put_nowait((identity, request))
        return jobs[identity]

    @app.get("/v1/videos/{identity}")
    async def get_video(identity: str):
        if identity not in jobs:
            raise HTTPException(404, "video job not found")
        return jobs[identity]

    @app.get("/v1/videos/{identity}/content")
    async def download_video(identity: str):
        job = await get_video(identity)
        if job["status"] != "completed":
            raise HTTPException(409, "video is not completed")
        return FileResponse(
            root / identity / "video.mp4",
            media_type="video/mp4",
            filename=identity + ".mp4",
        )

    return app


def serve(config, *, host, port, output_dir):
    import uvicorn

    uvicorn.run(create_app(config, output_dir), host=host, port=port)

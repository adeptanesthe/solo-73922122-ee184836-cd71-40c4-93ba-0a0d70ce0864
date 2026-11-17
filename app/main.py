"""FastAPI 后端：野生动物追踪串号校正与计数台。

会话保存在进程内存中（单用户工具，重启即清空）。
每次修改判断或参数都会使会话版本号 +1；求解任务记录启动时的版本，
前端据此丢弃“迟到”的旧结果，保证不覆盖新输入。
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .example import EXAMPLE_DESCRIPTION, build_example_detections, build_example_frames
from .solver import InputConflict, solve_tracking

MAX_FRAMES = 10
MAX_DETECTIONS = 30
MAX_JUDGMENTS = 200
MAX_VECTOR_DIM = 512
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# 默认参数针对内置交叉遮挡示例调优；上传自有数据时请按图像尺度与
# 外观向量的区分度调整（距离平方随图像尺寸增长很快，外观项有界）。
DEFAULT_PARAMS = {"w_dist": 1.0, "w_app": 10000.0, "penalty": 15000.0}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="野生动物追踪串号校正与计数台")

SESSIONS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------- 工具

def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


def _get_session(sid: str) -> dict:
    sess = SESSIONS.get(sid)
    if sess is None:
        _err(404, "会话不存在或已过期，请重新上传数据。")
    return sess


def _validate_vector(vec, ctx: str):
    if not isinstance(vec, list) or not vec:
        _err(400, f"{ctx}：外观向量必须是非空数组。")
    if len(vec) > MAX_VECTOR_DIM:
        _err(400, f"{ctx}：外观向量维度超过 {MAX_VECTOR_DIM}。")
    out = []
    for v in vec:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            _err(400, f"{ctx}：外观向量含有非法数值。")
        out.append(float(v))
    if math.sqrt(sum(v * v for v in out)) < 1e-12:
        _err(400, f"{ctx}：外观向量必须非零。")
    return out


def _validate_detections(payload, n_frames: int):
    """校验并规范化检测 JSON。接受 {"detections": [...]} 或直接 [...]。"""
    if isinstance(payload, dict):
        dets = payload.get("detections")
    else:
        dets = payload
    if not isinstance(dets, list):
        _err(400, "检测 JSON 必须是数组，或形如 {\"detections\": [...]} 的对象。")
    if len(dets) > MAX_DETECTIONS:
        _err(400, f"检测框数量 {len(dets)} 超过上限 {MAX_DETECTIONS}。")

    seen_ids = set()
    dim = None
    out = []
    for k, d in enumerate(dets):
        ctx = f"第 {k} 个检测框"
        if not isinstance(d, dict):
            _err(400, f"{ctx}：必须是对象。")
        bid = d.get("id")
        if not isinstance(bid, (str, int)) or isinstance(bid, bool) or str(bid) == "":
            _err(400, f"{ctx}：框号 id 必须是非空字符串或数字。")
        bid = str(bid)
        if bid in seen_ids:
            _err(400, f"框号 {bid} 重复。")
        seen_ids.add(bid)

        frame = d.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool):
            _err(400, f"框 {bid}：帧号 frame 必须是整数。")
        if not (0 <= frame < n_frames):
            _err(400, f"框 {bid}：帧号 {frame} 超出范围 [0, {n_frames - 1}]。")

        rect = d.get("rect")
        if not isinstance(rect, list) or len(rect) != 4:
            _err(400, f"框 {bid}：rect 必须是 [x, y, w, h] 四元数组。")
        rect_v = []
        for v in rect:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                _err(400, f"框 {bid}：rect 含有非法数值。")
            rect_v.append(float(v))
        if rect_v[2] <= 0 or rect_v[3] <= 0:
            _err(400, f"框 {bid}：rect 的宽、高必须为正数。")

        vec = _validate_vector(d.get("vector"), f"框 {bid}")
        if dim is None:
            dim = len(vec)
        elif len(vec) != dim:
            _err(400, f"框 {bid}：外观向量维度 {len(vec)} 与其他框（{dim}）不一致。")

        out.append({"id": bid, "frame": frame, "rect": rect_v, "vector": vec})
    out.sort(key=lambda d: (d["frame"], d["id"]))
    return out


def _new_session(frames, detections) -> dict:
    sid = uuid.uuid4().hex[:12]
    sess = {
        "id": sid,
        "version": 1,
        "frames": frames,  # [(content_type, bytes)]
        "detections": detections,
        "judgments": [],
        "params": dict(DEFAULT_PARAMS),
    }
    SESSIONS[sid] = sess
    return sess


def _payload(sess: dict) -> dict:
    return {
        "session_id": sess["id"],
        "version": sess["version"],
        "n_frames": len(sess["frames"]),
        "detections": sess["detections"],
        "judgments": sess["judgments"],
        "params": sess["params"],
        "limits": {"frames": MAX_FRAMES, "detections": MAX_DETECTIONS},
    }


# ---------------------------------------------------------------- 会话

@app.post("/api/sessions")
async def create_session(images: list[UploadFile] = File(...),
                         detections: UploadFile = File(...)):
    """上传至多 10 帧图片 + 30 框检测 JSON，创建会话。"""
    if not (1 <= len(images) <= MAX_FRAMES):
        _err(400, f"图片数量须为 1~{MAX_FRAMES}，收到 {len(images)}。")
    frames = []
    for img in images:
        ctype = img.content_type or ""
        if not ctype.startswith("image/"):
            _err(400, f"文件 {img.filename} 不是图片类型（{ctype or '未知'}）。")
        data = await img.read()
        if len(data) > MAX_IMAGE_BYTES:
            _err(400, f"图片 {img.filename} 超过大小限制。")
        frames.append((ctype, data))

    raw = await detections.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _err(400, f"检测 JSON 解析失败：{exc}")
    dets = _validate_detections(payload, len(frames))
    return _payload(_new_session(frames, dets))


@app.post("/api/example")
async def load_example():
    """创建内置交叉遮挡示例会话。"""
    sess = _new_session(build_example_frames(), build_example_detections())
    out = _payload(sess)
    out["description"] = EXAMPLE_DESCRIPTION
    return out


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    return _payload(_get_session(sid))


@app.get("/api/sessions/{sid}/frames/{idx}")
async def get_frame(sid: str, idx: int):
    sess = _get_session(sid)
    if not (0 <= idx < len(sess["frames"])):
        _err(404, "帧不存在。")
    ctype, data = sess["frames"][idx]
    return Response(content=data, media_type=ctype)


@app.get("/api/sessions/{sid}/detections.json")
async def download_detections(sid: str):
    sess = _get_session(sid)
    return JSONResponse({"detections": sess["detections"]})


# ---------------------------------------------------------------- 判断与参数

class Judgment(BaseModel):
    a: str
    b: str
    type: str = Field(pattern="^(same|different)$")


class JudgmentsBody(BaseModel):
    judgments: list[Judgment]


@app.put("/api/sessions/{sid}/judgments")
async def put_judgments(sid: str, body: JudgmentsBody):
    """整体替换身份判断；每次修改都会使旧求解结果作废（版本号 +1）。"""
    sess = _get_session(sid)
    if len(body.judgments) > MAX_JUDGMENTS:
        _err(400, f"判断数量超过上限 {MAX_JUDGMENTS}。")
    ids = {d["id"] for d in sess["detections"]}
    merged: dict[tuple, str] = {}
    for j in body.judgments:
        if j.a not in ids or j.b not in ids:
            _err(400, f"判断引用了不存在的框：{j.a} / {j.b}。")
        if j.a == j.b:
            _err(400, f"不能对同一个框（{j.a}）做同只/异只判断。")
        key = tuple(sorted((j.a, j.b)))
        merged[key] = j.type  # 同一对框以后到者为准
    sess["judgments"] = [
        {"a": a, "b": b, "type": t} for (a, b), t in merged.items()
    ]
    sess["version"] += 1
    return {"version": sess["version"], "judgments": body.judgments}


class ParamsBody(BaseModel):
    w_dist: float = Field(ge=0)
    w_app: float = Field(ge=0)
    penalty: float = Field(ge=0)


@app.put("/api/sessions/{sid}/params")
async def put_params(sid: str, body: ParamsBody):
    sess = _get_session(sid)
    for name, v in (("w_dist", body.w_dist), ("w_app", body.w_app), ("penalty", body.penalty)):
        if not math.isfinite(v):
            _err(400, f"参数 {name} 必须是有限非负数。")
    sess["params"] = {"w_dist": body.w_dist, "w_app": body.w_app, "penalty": body.penalty}
    sess["version"] += 1
    return {"version": sess["version"], "params": sess["params"]}


# ---------------------------------------------------------------- 求解

def _run_job(job_id: str, snapshot: dict):
    try:
        # 求解器使用下标引用，这里把框号映射为下标
        id_to_idx = {d["id"]: i for i, d in enumerate(snapshot["detections"])}
        judgments = [
            {"a": id_to_idx[j["a"]], "b": id_to_idx[j["b"]], "type": j["type"]}
            for j in snapshot["judgments"]
        ]
        result = solve_tracking(
            snapshot["detections"],
            judgments,
            snapshot["params"]["w_dist"],
            snapshot["params"]["w_app"],
            snapshot["params"]["penalty"],
        )
        JOBS[job_id].update(status="done", result=result)
    except InputConflict as exc:
        JOBS[job_id].update(
            status="done",
            result={"feasible": False, "message": f"无解：{exc} 不提供动物计数。"},
        )
    except Exception as exc:  # noqa: BLE001 - 任何求解异常都不能拖垮服务
        JOBS[job_id].update(status="error", error=f"求解内部错误：{exc}")


@app.post("/api/sessions/{sid}/solve")
async def start_solve(sid: str):
    """启动异步求解。任务记录当前输入版本，前端据此丢弃迟到结果。"""
    sess = _get_session(sid)
    job_id = uuid.uuid4().hex[:12]
    snapshot = {
        "detections": copy.deepcopy(sess["detections"]),
        "judgments": copy.deepcopy(sess["judgments"]),
        "params": dict(sess["params"]),
    }
    JOBS[job_id] = {"status": "running", "version": sess["version"], "result": None, "error": None}
    loop = asyncio.get_running_loop()
    loop.run_in_executor(EXECUTOR, _run_job, job_id, snapshot)
    return {"job_id": job_id, "version": sess["version"]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        _err(404, "求解任务不存在。")
    return job


# ---------------------------------------------------------------- 页面

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

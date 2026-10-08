"""Small same-origin HTTP surface; all domain operations use the facade."""
from __future__ import annotations

from collections import OrderedDict
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator
from starlette.exceptions import HTTPException

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.interactive_schemas import ClientError, TurnView
from agent.application.local_resources import RegisteredInput, ResourceAdmissionError
from agent.application.uploads import H5ADUploadAdmission, UploadError

from .config import ScientificInputSet


logger = logging.getLogger(__name__)
STATIC_ROOT = Path(__file__).with_name('static')


class _Body(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class CreateSession(_Body):
    session_id: StrictStr | None = Field(default=None, min_length=1, max_length=256, pattern=r'^[^/\\]+$')

    @field_validator('session_id')
    @classmethod
    def route_identity(cls, value):
        return _route_identity(value)


class SubmitTurn(_Body):
    turn_id: StrictStr = Field(min_length=1, max_length=256, pattern=r'^[^/\\]+$')
    expected_generation: StrictInt = Field(ge=0)
    utterance: StrictStr = Field(min_length=1, max_length=4096)
    profile_id: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    input_set_id: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    resource_id: StrictStr | None = Field(default=None, pattern=r'^local-[0-9a-f]{64}$')
    predecessor_turn_id: StrictStr | None = Field(default=None, min_length=1, max_length=256)

    @field_validator('turn_id')
    @classmethod
    def route_identity(cls, value):
        return _route_identity(value)


class ActivateRevision(_Body):
    turn_id: StrictStr = Field(min_length=1, max_length=256, pattern=r'^[^/\\]+$')
    revision_id: StrictStr = Field(min_length=1, max_length=256)
    expected_generation: StrictInt = Field(ge=0)

    @field_validator('turn_id')
    @classmethod
    def route_identity(cls, value):
        return _route_identity(value)


def _route_identity(value):
    if value in {'.', '..'}:
        raise ValueError('URL dot segments cannot identify a session or turn.')
    return value


def _error(code, message, status):
    return JSONResponse({'error': ClientError(code, message).to_dict()}, status_code=status)


def _status_code(code):
    return {
        'INTERACTIVE_SESSION_INVALID': 404,
        'INTERACTIVE_REFERENCE_INVALID': 404,
        'INTERACTIVE_ARTIFACT_UNAVAILABLE': 404,
        'INTERACTIVE_TURN_CONFLICT': 409,
        'INTERACTIVE_GENERATION_CONFLICT': 409,
        'INTERACTIVE_OPERATION_ACTIVE': 409,
        'INTERACTIVE_MODEL_UNAVAILABLE': 400,
        'INTERACTIVE_APPLICATION_FAILED': 500,
    }.get(code, 400)


@dataclass
class _Receipt:
    fingerprint: str
    submitted: TurnView
    future: Future | None = None


class _LocalWorkers:
    """Bounded active calls, never a durable queue or source of run status.

    Receipts cover the short pre-capture interval and pre-capture failures only.
    A server restart loses receipts, but never replays a call. Captured turns and
    all scientific checkpoints remain exclusively application-owned.
    """

    def __init__(self, application, max_workers):
        self.application = application
        self.max_workers = max_workers
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='agent-turn')
        self.lock = Lock()
        self.receipts = OrderedDict()
        self.closed = False

    def _invoke(self, session_id, payload, inputs):
        try:
            return self.application.submit_turn(session_id, payload.turn_id, payload.utterance,
                expected_generation=payload.expected_generation, profile_id=payload.profile_id,
                predecessor_turn_id=payload.predecessor_turn_id, **self._input_arguments(inputs))
        except InteractiveBoundaryError as exc:
            logger.warning('Interactive turn failed: %s', exc.error.code, exc_info=True)
            return exc.error
        except ResourceAdmissionError as exc:
            logger.warning('Registered input admission failed: %s', exc.code)
            return exc.error
        except Exception:
            logger.exception('Unexpected interactive turn failure')
            return ClientError('INTERACTIVE_APPLICATION_FAILED',
                               'The application could not complete this turn.')

    @staticmethod
    def _input_arguments(inputs):
        return ({'registered_input': inputs} if isinstance(inputs, RegisteredInput)
                else {'execution_inputs': inputs})

    def submit(self, session_id, payload, inputs):
        # Domain validation and normalization belong to M18.2, not HTTP models.
        fingerprint = self.application.validate_submission(session_id, payload.turn_id, payload.utterance,
            expected_generation=payload.expected_generation, profile_id=payload.profile_id,
            predecessor_turn_id=payload.predecessor_turn_id, **self._input_arguments(inputs))
        key = (session_id, payload.turn_id)
        with self.lock:
            receipt = self.receipts.get(key)
            if receipt is not None:
                if receipt.fingerprint != fingerprint:
                    raise InteractiveBoundaryError('INTERACTIVE_TURN_CONFLICT')
                return
            # A completed/pending captured retry is an application read, never a
            # new worker. M18.2 validation already checked its full exact identity.
            try:
                self.application.turn(session_id, payload.turn_id)
            except InteractiveBoundaryError as exc:
                if exc.error.code != 'INTERACTIVE_REFERENCE_INVALID':
                    raise
            else:
                return
            active = sum(r.future is None or not r.future.done() for r in self.receipts.values())
            if self.closed or active >= self.max_workers:
                raise _WorkersBusy()
            while len(self.receipts) >= 128:
                oldest = next((k for k, r in self.receipts.items()
                               if r.future is not None and r.future.done()), None)
                if oldest is None:
                    raise _WorkersBusy()
                del self.receipts[oldest]
            receipt = _Receipt(fingerprint, TurnView(session_id, payload.turn_id,
                payload.utterance, payload.expected_generation, profile_id=payload.profile_id,
                status='submitted'))
            self.receipts[key] = receipt
            try:
                receipt.future = self.pool.submit(self._invoke, session_id, payload, inputs)
            except Exception:
                del self.receipts[key]
                raise

    def status(self, session_id, turn_id):
        with self.lock:
            receipt = self.receipts.get((session_id, turn_id))
        failure = (receipt.future.result() if receipt is not None and receipt.future is not None
                   and receipt.future.done() else None)
        try:
            view = self.application.status(session_id, turn_id)
        except InteractiveBoundaryError as exc:
            if exc.error.code != 'INTERACTIVE_REFERENCE_INVALID' or receipt is None:
                raise
            if isinstance(failure, ClientError):
                return replace(receipt.submitted, status='failed', error=failure)
            if isinstance(failure, TurnView):
                return failure
            return receipt.submitted
        # Retain a durable checkpoint even when transport processing failed.
        return replace(view, error=failure) if isinstance(failure, ClientError) and view.error is None else view

    def close(self):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True)


class _WorkersBusy(Exception):
    pass


def create_app(application: InteractiveAgentApplication, *, input_sets=(), max_workers=2, uploads=None):
    """Compose a single lab-host server around one configured domain facade."""
    if not isinstance(application, InteractiveAgentApplication):
        raise TypeError('A configured InteractiveAgentApplication is required.')
    if type(max_workers) is not int or not 1 <= max_workers <= 8:
        raise ValueError('max_workers must be between 1 and 8.')
    if uploads is not None and not isinstance(uploads, H5ADUploadAdmission):
        raise TypeError('uploads must use the application H5AD upload boundary.')
    if uploads is not None and uploads.resources is not application.resources:
        raise ValueError('uploads must use this application resource owner.')
    if not isinstance(input_sets, tuple) or not all(isinstance(s, ScientificInputSet) for s in input_sets):
        raise TypeError('input_sets must be an operator-configured tuple.')
    inputs_by_id = {s.input_set_id: s for s in input_sets}
    if len(inputs_by_id) != len(input_sets):
        raise ValueError('Duplicate scientific input-set identifier.')
    workers = _LocalWorkers(application, max_workers)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            # Graceful shutdown drains calls; browser disconnect is unrelated.
            await asyncio.to_thread(workers.close)

    app = FastAPI(title='Agent interactive application', docs_url=None,
                  redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.middleware('http')
    async def same_origin(request, call_next):
        if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            origin = request.headers.get('origin')
            if origin is not None and origin != str(request.base_url).rstrip('/'):
                return _error('WEB_ORIGIN_INVALID', 'Use the same-origin Agent application.', 403)
            content_type = request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
            if (request.url.path == '/api/v1/sessions'
                    or (request.url.path.startswith('/api/v1/sessions/')
                        and request.url.path.endswith(('/turns', '/activate')))):
                if content_type != 'application/json':
                    return _error('WEB_REQUEST_INVALID', 'Send a valid JSON request.', 415)
                # Browser input contains text/identities only, never scientific data.
                if len(await request.body()) > 32768:
                    return _error('WEB_REQUEST_INVALID', 'The request is too large.', 413)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        return response

    @app.exception_handler(InteractiveBoundaryError)
    async def boundary_error(request, exc):
        return JSONResponse({'error': exc.error.to_dict()}, status_code=_status_code(exc.error.code))

    @app.exception_handler(ResourceAdmissionError)
    async def resource_error(request, exc):
        status = 404 if exc.code == 'LOCAL_RESOURCE_UNAVAILABLE' else 400
        return JSONResponse({'error': exc.error.to_dict()}, status_code=status)

    @app.exception_handler(UploadError)
    async def upload_error(request, exc):
        return _error(exc.code, exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return _error('WEB_REQUEST_INVALID', 'The HTTP request is invalid.', 422)

    @app.exception_handler(_WorkersBusy)
    async def busy(request, exc):
        response = _error('WEB_WORKERS_BUSY', 'All local turn workers are busy. Retry this submission shortly.', 503)
        response.headers['Retry-After'] = '2'
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return _error('WEB_REQUEST_INVALID', 'The requested HTTP operation is unavailable.', exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.exception('Unexpected HTTP operation failure', exc_info=exc)
        return _error('INTERACTIVE_APPLICATION_FAILED', 'The application could not complete this request.', 500)

    @app.get('/api/v1/health')
    def health():
        return {'status': 'ok'}

    @app.get('/api/v1/models')
    def models():
        return {'choices': [c.to_dict() for c in application.model_choices()]}

    @app.get('/api/v1/input-sets')
    def scientific_inputs():
        return {'choices': [s.choice() for s in input_sets]}

    @app.get('/api/v1/resources')
    def uploaded_resources():
        return {'enabled': uploads is not None,
                'choices': [] if uploads is None else list(uploads.choices())}

    @app.put('/api/v1/uploads/{upload_id}', status_code=201)
    async def upload_h5ad(upload_id: str, request: Request,
                         filename: str = Query(min_length=1, max_length=160)):
        if uploads is None:
            return _error('WEB_UPLOAD_UNAVAILABLE', 'H5AD attachments are not configured.', 404)
        if list(request.query_params.keys()) != ['filename'] or len(request.query_params.getlist('filename')) != 1:
            return _error('WEB_UPLOAD_REQUEST_INVALID', 'The upload request is invalid.', 422)
        content_type = request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
        if content_type != 'application/octet-stream':
            return _error('WEB_UPLOAD_REQUEST_INVALID', 'Send the file as a binary stream.', 415)
        length = request.headers.get('content-length')
        expected_size = None
        if length is not None:
            if (len(request.headers.getlist('content-length')) != 1
                    or not length.isascii() or not length.isdecimal()):
                return _error('WEB_UPLOAD_REQUEST_INVALID', 'The upload request is invalid.', 422)
            normalized_length = length.lstrip('0') or '0'
            if len(normalized_length) > 20 or int(normalized_length) > uploads.max_bytes:
                return _error('UPLOAD_TOO_LARGE', 'The file exceeds the configured attachment size limit.', 413)
            expected_size = int(normalized_length)
        record = await uploads.receive(upload_id, filename, request.stream(),
                                       expected_size=expected_size)
        return record.public()

    @app.post('/api/v1/sessions', status_code=201)
    def create_session(body: CreateSession):
        return application.create_session(body.session_id or str(uuid4())).to_dict()

    @app.get('/api/v1/sessions/{session_id}')
    def reopen_session(session_id: str):
        return application.reopen_session(session_id).to_dict()

    @app.get('/api/v1/sessions/{session_id}/revisions/{revision_id}')
    def revision(session_id: str, revision_id: str):
        return application.revision(session_id, revision_id).to_dict()

    @app.post('/api/v1/sessions/{session_id}/activate')
    def activate_revision(session_id: str, body: ActivateRevision):
        return application.activate_revision(session_id, body.turn_id, body.revision_id,
            expected_generation=body.expected_generation).to_dict()

    @app.get('/api/v1/sessions/{session_id}/revisions/{revision_id}/evidence')
    def evidence(session_id: str, revision_id: str,
                 output_name: str = Query(min_length=1, max_length=256),
                 fields: list[str] | None = Query(default=None, max_length=128),
                 detail_section: str | None = Query(default=None, max_length=128),
                 subject: str | None = Query(default=None, max_length=128),
                 limit: int = Query(default=8, ge=1, le=32),
                 candidate: str | None = Query(default=None, max_length=128)):
        return application.evidence(session_id, revision_id, output_name,
            fields=None if fields is None else tuple(fields), detail_section=detail_section,
            subject=subject, limit=limit, candidate=candidate).to_dict()

    @app.get('/api/v1/sessions/{session_id}/revisions/{revision_id}/artifacts')
    def artifacts(session_id: str, revision_id: str):
        return {'artifacts': [a.to_dict() for a in application.artifact_handles(session_id, revision_id)]}

    @app.get('/api/v1/sessions/{session_id}/revisions/{revision_id}/artifacts/{handle}')
    def artifact(session_id: str, revision_id: str, handle: str, download: bool = False):
        content = application.artifact_content(session_id, revision_id, handle)
        disposition = 'attachment' if download else 'inline'
        return Response(content=content.data, media_type=content.content_type, headers={
            'Content-Disposition': f'{disposition}; filename="{content.filename}"',
            'X-Artifact-Source-SHA256': content.reference.sha256,
            'X-Artifact-Content-SHA256': content.content_sha256,
            'X-Artifact-Presentation': content.presentation,
        })

    @app.post('/api/v1/sessions/{session_id}/turns', status_code=202)
    def submit_turn(session_id: str, body: SubmitTurn):
        inputs = {}
        if body.resource_id is not None:
            if body.input_set_id is not None or uploads is None:
                raise InteractiveBoundaryError('INTERACTIVE_INPUT_INVALID')
            # Completed retries remain readable even after source bytes disappear.
            # The facade validates bytes itself before each new consumption.
            inputs = uploads.resolve(body.resource_id, verify_source=False)
        if body.input_set_id is not None:
            input_set = inputs_by_id.get(body.input_set_id)
            if input_set is None:
                raise InteractiveBoundaryError('INTERACTIVE_INPUT_INVALID')
            inputs = input_set.inputs()
        workers.submit(session_id, body, inputs)
        return {'session_id': session_id, 'turn_id': body.turn_id}

    @app.get('/api/v1/sessions/{session_id}/turns/{turn_id}')
    @app.get('/api/v1/sessions/{session_id}/turns/{turn_id}/status')
    def turn(session_id: str, turn_id: str):
        return workers.status(session_id, turn_id).to_dict()

    @app.post('/api/v1/sessions/{session_id}/turns/{turn_id}/cancel')
    def cancel(session_id: str, turn_id: str):
        application.cancel_turn(session_id, turn_id)
        return application.status(session_id, turn_id).to_dict()

    @app.get('/')
    def index():
        return FileResponse(STATIC_ROOT / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC_ROOT), name='static')
    return app


__all__ = ['create_app']

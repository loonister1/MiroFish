"""
Zep Cloud stand-in backed by Graphiti on a local Neo4j.

Every MiroFish service reaches graph memory through get_zep_client(), so this serves the slice of
the zep_cloud client they call and the services keep their Zep-shaped code. Zep Cloud itself runs
on Graphiti, so nodes, edges, temporal fields and hybrid search carry over one to one.

Ingestion keeps Zep's contract: add() and batch.process() return at once, and a background worker
extracts entities with Gemini and reports `processed` per episode. Episodes waiting for the same
graph are ingested together through add_episode_bulk, about 3x faster than one at a time.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Callable, Optional

import httpx
from google.genai import errors as genai_errors

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")  # read when graphiti_core loads

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.errors import GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from pydantic import Field, create_model
from zep_cloud import NotFoundError

from ..config import Config
from .atomic_json import write_json_atomic
from .logger import get_logger

logger = get_logger("mirofish.graphiti")

BULK_SIZE = 20
EMBEDDING_MODEL = "gemini-embedding-001"
_GRAPH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


PASSING_ATTEMPTS = 6  # waits 10+20+40+80+120s, about 4.5 minutes in all
FAILURE_ATTEMPTS = 3


def _passes_with_time(error: BaseException) -> bool:
    """A 429, a Gemini overload (5xx) or a network drop: these clear up if we wait."""

    for cause in (error, error.__cause__):
        if isinstance(cause, (RateLimitError, httpx.TransportError)):
            return True
        if isinstance(cause, genai_errors.APIError) and (cause.code == 429 or cause.code >= 500):
            return True
    return False


async def _patiently(call: Callable[[], Any]) -> Any:
    """One failed Gemini call otherwise fails a whole batch of episodes, and with it the run."""

    waited = failed = 0
    while True:
        try:
            return await call()
        except Exception as error:
            if _passes_with_time(error):
                waited += 1
                if waited == PASSING_ATTEMPTS:
                    raise
                await asyncio.sleep(min(10 * 2 ** (waited - 1), 120))
            else:  # e.g. Gemini looping on a JSON string until it is cut off; a fresh try usually works
                failed += 1
                if failed == FAILURE_ATTEMPTS:
                    raise
                await asyncio.sleep(10 * failed)


class PatientGeminiClient(GeminiClient):
    """GeminiClient gives up at once on a 429 and after two tries on anything else."""

    FAILURE_ATTEMPTS = FAILURE_ATTEMPTS

    async def generate_response(self, messages, *args, **kwargs):
        parent = super().generate_response
        # Fresh copies: GeminiClient appends its notes to the messages it is given.
        return await _patiently(
            lambda: parent([message.model_copy() for message in messages], *args, **kwargs)
        )


class PatientGeminiEmbedder(GeminiEmbedder):
    """GeminiEmbedder never retries, so a single DNS blip failed a run's last batch."""

    async def create(self, input_data):
        return await _patiently(lambda: GeminiEmbedder.create(self, input_data))

    async def create_batch(self, input_data_list):
        return await _patiently(lambda: GeminiEmbedder.create_batch(self, input_data_list))


def _not_found(what: str) -> NotFoundError:
    return NotFoundError(body={"message": f"{what} not found"})


def _zep_shaped(item: Any) -> SimpleNamespace:
    """A Graphiti node/edge with Zep's `uuid_` name and without the embedding vectors."""

    data = item.model_dump(exclude={"name_embedding", "fact_embedding"})
    return SimpleNamespace(**data, uuid_=item.uuid)


def _ontology_spec(entities: dict | None, edges: dict | None) -> dict:
    """Plain-JSON form of the zep_cloud ontology classes, so it outlives the process."""

    def fields(model) -> dict:
        return {name: field.description or "" for name, field in model.model_fields.items()}

    return {
        "entities": {
            name: {"description": model.__doc__ or "", "fields": fields(model)}
            for name, model in (entities or {}).items()
        },
        "edges": {
            name: {
                "description": model.__doc__ or "",
                "fields": fields(model),
                "source_targets": [[st.source, st.target] for st in source_targets],
            }
            for name, (model, source_targets) in (edges or {}).items()
        },
    }


def _graphiti_ontology(spec: dict | None) -> dict:
    """add_episode_bulk keyword arguments for a stored ontology spec."""

    if not spec:
        return {}

    def model(name: str, definition: dict):
        return create_model(
            name,
            __doc__=definition["description"],
            **{
                field: (Optional[str], Field(default=None, description=description))
                for field, description in definition["fields"].items()
            },
        )

    edge_type_map: dict[tuple[str, str], list[str]] = {}
    for name, definition in spec["edges"].items():
        for source, target in definition["source_targets"]:
            edge_type_map.setdefault((source, target), []).append(name)
    return {
        "entity_types": {n: model(n, d) for n, d in spec["entities"].items()} or None,
        "edge_types": {n: model(n, d) for n, d in spec["edges"].items()} or None,
        "edge_type_map": edge_type_map or None,
    }


class GraphitiZepClient:
    """The zep_cloud client surface MiroFish uses (graph.*, batch.*), served by Graphiti."""

    def __init__(self, make_graphiti: Callable[[], Graphiti], meta_dir: str):
        self._meta_dir = meta_dir
        os.makedirs(meta_dir, exist_ok=True)
        self._episodes: dict[str, SimpleNamespace] = {}
        self._pending: list[SimpleNamespace] = []  # touched only on the loop thread
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="graphiti", daemon=True).start()
        self._run(self._start(make_graphiti))
        self.graph = _Graph(self)
        self.batch = _Batches(self)

    def _run(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    async def _start(self, make_graphiti) -> None:
        # Built on the loop so Graphiti's async Neo4j and Gemini clients bind to it.
        self._graphiti = make_graphiti()
        await self._graphiti.build_indices_and_constraints()
        self._wakeup = asyncio.Event()
        self._worker = asyncio.get_running_loop().create_task(self._ingest_forever())

    # Graph metadata (name, ontology) lives beside the uploads; the graph itself lives in Neo4j.
    def _meta_path(self, graph_id: str) -> str:
        if not isinstance(graph_id, str) or not _GRAPH_ID_PATTERN.match(graph_id):
            raise _not_found(f"Graph {graph_id!r}")
        return os.path.join(self._meta_dir, f"{graph_id}.json")

    def _read_meta(self, graph_id: str) -> dict:
        try:
            with open(self._meta_path(graph_id), encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise _not_found(f"Graph {graph_id}") from None

    def _write_meta(self, graph_id: str, meta: dict) -> None:
        write_json_atomic(self._meta_path(graph_id), meta)

    def _new_episode(self, graph_id: str, data: str, source_description: str | None,
                     created_at: str | None) -> SimpleNamespace:
        self._meta_path(graph_id)  # reject unsafe ids before queueing
        episode = SimpleNamespace(
            uuid_=uuid.uuid4().hex,
            graph_id=graph_id,
            data=data,
            source_description=source_description or "MiroFish",
            reference_time=(
                datetime.fromisoformat(created_at) if created_at else datetime.now(timezone.utc)
            ),
            processed=False,
            error=None,
        )
        self._episodes[episode.uuid_] = episode
        return episode

    def _enqueue(self, episodes: list[SimpleNamespace]) -> None:
        def push():
            self._pending.extend(episodes)
            self._wakeup.set()

        self._loop.call_soon_threadsafe(push)

    async def _ingest_forever(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            while self._pending:
                graph_id = self._pending[0].graph_id
                episodes = [e for e in self._pending if e.graph_id == graph_id][:BULK_SIZE]
                for episode in episodes:
                    self._pending.remove(episode)
                await self._ingest(graph_id, episodes)

    async def _ingest(self, graph_id: str, episodes: list[SimpleNamespace]) -> None:
        try:
            ontology = _graphiti_ontology(self._read_meta(graph_id).get("ontology"))
            # PatientGeminiClient rides out Gemini rate limits; any other failure fails the batch.
            await self._graphiti.add_episode_bulk(
                [
                    RawEpisode(
                        name=episode.source_description,
                        content=episode.data,
                        source_description=episode.source_description,
                        source=EpisodeType.text,
                        reference_time=episode.reference_time,
                    )
                    for episode in episodes
                ],
                group_id=graph_id,
                **ontology,
            )
        except Exception as error:
            logger.error(f"Graphiti ingestion of {len(episodes)} episode(s) into {graph_id} failed: {error!r} (cause: {error.__cause__!r})")
            for episode in episodes:
                # GeminiClient raises a bare Exception whose cause holds the reason.
                episode.error = str(error) or str(error.__cause__ or "") or type(error).__name__
        for episode in episodes:
            episode.processed = True


class _Graph:
    def __init__(self, client: GraphitiZepClient):
        self._client = client
        self.node = _Nodes(client)
        self.edge = _Edges(client)
        self.episode = _Episodes(client)

    def create(self, graph_id: str, name: str | None = None, description: str | None = None, **_):
        self._client._write_meta(
            graph_id, {"graph_id": graph_id, "name": name, "description": description, "ontology": None}
        )
        return self.get(graph_id)

    def get(self, graph_id: str):
        meta = self._client._read_meta(graph_id)
        return SimpleNamespace(graph_id=graph_id, name=meta.get("name"), description=meta.get("description"))

    def delete(self, graph_id: str):
        path = self._client._meta_path(graph_id)
        self._client._run(clear_data(self._client._graphiti.driver, group_ids=[graph_id]))
        if os.path.exists(path):
            os.remove(path)

    def set_ontology(self, graph_ids: list[str], entities: dict | None = None,
                     edges: dict | None = None, **_):
        spec = _ontology_spec(entities, edges)
        for graph_id in graph_ids:
            meta = self._client._read_meta(graph_id)
            self._client._write_meta(graph_id, {**meta, "ontology": spec})

    def add(self, graph_id: str, data: str, type: str = "text", created_at: str | None = None,
            source_description: str | None = None, **_):
        episode = self._client._new_episode(graph_id, data, source_description, created_at)
        self._client._enqueue([episode])
        return SimpleNamespace(uuid_=episode.uuid_, processed=False)

    def search(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges", **_):
        # Every search ranks with RRF: Graphiti's Gemini cross-encoder costs one LLM call per candidate.
        recipe = NODE_HYBRID_SEARCH_RRF if scope == "nodes" else EDGE_HYBRID_SEARCH_RRF
        results = self._client._run(self._client._graphiti.search_(
            query, config=recipe.model_copy(update={"limit": limit}), group_ids=[graph_id],
        ))
        return SimpleNamespace(
            edges=[_zep_shaped(edge) for edge in results.edges],
            nodes=[_zep_shaped(node) for node in results.nodes],
        )


class _Listing:
    """Zep's `with_raw_response.get_by_graph_id` page shape for the paging helper."""

    def __init__(self, fetch: Callable[[str], list]):
        self._fetch = fetch
        self.with_raw_response = self

    def get_by_graph_id(self, graph_id: str, limit: int | None = None, cursor: str | None = None):
        # shortcut: returns the whole graph as one page, fine for a local DB with no rate limits;
        # switch to Graphiti's uuid_cursor paging if graphs outgrow memory.
        return SimpleNamespace(data=self._fetch(graph_id), headers={})


class _Nodes(_Listing):
    def __init__(self, client: GraphitiZepClient):
        self._client = client
        super().__init__(lambda graph_id: [
            _zep_shaped(node)
            for node in client._run(EntityNode.get_by_group_ids(client._graphiti.driver, [graph_id]))
        ])

    def get(self, uuid_: str):
        try:
            return _zep_shaped(self._client._run(EntityNode.get_by_uuid(self._client._graphiti.driver, uuid_)))
        except NodeNotFoundError:
            raise _not_found(f"Node {uuid_}") from None

    def get_edges(self, node_uuid: str):
        edges = self._client._run(EntityEdge.get_by_node_uuid(self._client._graphiti.driver, node_uuid))
        return [_zep_shaped(edge) for edge in edges]


class _Edges(_Listing):
    def __init__(self, client: GraphitiZepClient):
        async def fetch(graph_id: str) -> list:
            try:
                return await EntityEdge.get_by_group_ids(client._graphiti.driver, [graph_id])
            except GroupsEdgesNotFoundError:  # Zep lists an edgeless graph as empty
                return []

        super().__init__(lambda graph_id: [_zep_shaped(edge) for edge in client._run(fetch(graph_id))])


class _Episodes:
    def __init__(self, client: GraphitiZepClient):
        self._client = client

    def get(self, uuid_: str):
        episode = self._client._episodes.get(uuid_)
        if episode is None:
            raise _not_found(f"Episode {uuid_}")
        if episode.error:
            raise RuntimeError(f"Graphiti ingestion failed: {episode.error}")
        return SimpleNamespace(uuid_=uuid_, processed=episode.processed)


class _Batches:
    """Zep's Batch API: items queue as a draft, then ingest in order once processed."""

    def __init__(self, client: GraphitiZepClient):
        self._client = client
        self._batches: dict[str, SimpleNamespace] = {}

    def _get(self, batch_id: str) -> SimpleNamespace:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise _not_found(f"Batch {batch_id}")
        return batch

    def create(self, metadata: dict | None = None, **_):
        batch = SimpleNamespace(batch_id=uuid.uuid4().hex, metadata=metadata or {}, episodes=[], submitted=False)
        self._batches[batch.batch_id] = batch
        return batch

    def add(self, batch_id: str, items: list, **_):
        batch = self._get(batch_id)
        start = len(batch.episodes)
        batch.episodes.extend(
            self._client._new_episode(item.graph_id, item.data, item.source_description,
                                      getattr(item, "created_at", None))
            for item in items
        )
        return self.list_items(batch_id).items[start:]

    def process(self, batch_id: str, **_):
        batch = self._get(batch_id)
        if not batch.submitted:
            batch.submitted = True
            self._client._enqueue(list(batch.episodes))
        return self.get(batch_id)

    def get(self, batch_id: str, **_):
        batch = self._get(batch_id)
        total = len(batch.episodes)
        done = [e for e in batch.episodes if e.processed]
        succeeded = sum(1 for e in done if not e.error)
        if not batch.submitted:
            status = "draft"
        elif len(done) < total:
            status = "processing"
        elif succeeded == total:
            status = "succeeded"
        else:
            status = "partial" if succeeded else "failed"
        return SimpleNamespace(
            batch_id=batch_id,
            metadata=batch.metadata,
            status=status,
            progress=SimpleNamespace(
                percent_complete=100 * len(done) / total if total else 100,
                succeeded_items=succeeded,
            ),
        )

    def list_items(self, batch_id: str, limit: int | None = None, cursor: Any = None, **_):
        items = [
            SimpleNamespace(
                sequence_index=index,
                episode_uuid=episode.uuid_,
                source_uuid=None,
                status=("failed" if episode.error else "succeeded") if episode.processed else "pending",
                error=episode.error,
            )
            for index, episode in enumerate(self._get(batch_id).episodes)
        ]
        return SimpleNamespace(items=items, next_cursor=None)

    def list(self, limit: int | None = None, cursor: Any = None, **_):
        return SimpleNamespace(batches=[self.get(b) for b in self._batches], next_cursor=None)


@lru_cache(maxsize=1)
def get_graphiti_client() -> GraphitiZepClient:
    def make_graphiti() -> Graphiti:
        # shortcut: assumes LLM_API_KEY is a Gemini key (this install's setup); use Graphiti's
        # matching LLM/embedder clients if MiroFish moves to another provider.
        llm = LLMConfig(
            api_key=Config.LLM_API_KEY, model=Config.LLM_MODEL_NAME, small_model=Config.LLM_MODEL_NAME
        )
        return Graphiti(
            Config.NEO4J_URI,
            Config.NEO4J_USER,
            Config.NEO4J_PASSWORD,
            llm_client=PatientGeminiClient(llm),
            embedder=PatientGeminiEmbedder(
                GeminiEmbedderConfig(api_key=Config.LLM_API_KEY, embedding_model=EMBEDDING_MODEL)
            ),
            cross_encoder=GeminiRerankerClient(llm),  # Graphiti requires one; searches use RRF
        )

    return GraphitiZepClient(make_graphiti, os.path.join(Config.UPLOAD_FOLDER, "graphiti_graphs"))

import asyncio
import threading
import time

import pytest
from graphiti_core.nodes import EntityNode
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_config import EdgeReranker
from zep_cloud import BatchAddItem, NotFoundError

from app.services import graph_builder as graph_builder_module
from app.services.graph_builder import GraphBuilderService
from app.utils import graphiti_zep
from app.utils.graphiti_zep import GraphitiZepClient
from app.utils.zep_paging import fetch_all_edges, fetch_all_nodes


class FakeGraphiti:
    def __init__(self, fail=False, gate=None):
        self.driver = object()
        self.bulk_calls = []
        self.searches = []
        self.fail = fail
        self.gate = gate
        self.started = threading.Event()

    async def build_indices_and_constraints(self):
        pass

    async def add_episode_bulk(self, bulk_episodes, group_id, **ontology):
        self.started.set()
        if self.gate is not None:
            await asyncio.to_thread(self.gate.wait)
        self.bulk_calls.append((group_id, [e.content for e in bulk_episodes], ontology))
        if self.fail:
            raise RuntimeError("gemini down")

    async def search_(self, query, config, group_ids):
        self.searches.append((query, config, group_ids))
        node = EntityNode(name="Ardila Sports", group_id=group_ids[0], labels=["Entity"])
        return type("Results", (), {"nodes": [node], "edges": []})()


def _client(tmp_path, graphiti):
    return GraphitiZepClient(lambda: graphiti, str(tmp_path))


def _wait_processed(client, episode_uuid):
    deadline = time.time() + 5
    while not client.graph.episode.get(uuid_=episode_uuid).processed:
        assert time.time() < deadline, "episode was never ingested"
        time.sleep(0.01)


def test_added_episode_is_queued_then_ingested_into_its_graph(tmp_path):
    graphiti = FakeGraphiti()
    client = _client(tmp_path, graphiti)
    client.graph.create(graph_id="g1", name="Padel")

    episode = client.graph.add(
        graph_id="g1", type="text", data="Agent posted: padel is booming",
        created_at="2026-10-14T10:00:00+05:00", source_description="sim batch",
    )
    _wait_processed(client, episode.uuid_)

    assert [(group, texts) for group, texts, _ in graphiti.bulk_calls] == [
        ("g1", ["Agent posted: padel is booming"])
    ]


def test_queued_episodes_of_a_graph_are_ingested_together_in_send_order(tmp_path):
    gate = threading.Event()
    graphiti = FakeGraphiti(gate=gate)
    client = _client(tmp_path, graphiti)
    for graph_id in ("a", "b"):
        client.graph.create(graph_id=graph_id, name=graph_id)

    first = client.graph.add(graph_id="a", type="text", data="a1")
    assert graphiti.started.wait(5)  # the worker now holds "a1" behind the gate
    later = [
        client.graph.add(graph_id=graph_id, type="text", data=data)
        for graph_id, data in (("a", "a2"), ("b", "b1"), ("a", "a3"))
    ]
    gate.set()
    for episode in [first, *later]:
        _wait_processed(client, episode.uuid_)

    assert [(group, texts) for group, texts, _ in graphiti.bulk_calls] == [
        ("a", ["a1"]),
        ("a", ["a2", "a3"]),
        ("b", ["b1"]),
    ]


def test_failed_ingestion_surfaces_on_the_episode_instead_of_passing_silently(tmp_path):
    client = _client(tmp_path, FakeGraphiti(fail=True))
    client.graph.create(graph_id="g1", name="Padel")
    episode = client.graph.add(graph_id="g1", type="text", data="lost")

    deadline = time.time() + 5
    while True:
        try:
            client.graph.episode.get(uuid_=episode.uuid_)
        except RuntimeError as error:
            assert "gemini down" in str(error)
            break
        assert time.time() < deadline, "failure was never reported"
        time.sleep(0.01)


def test_batch_api_reports_progress_and_items_like_zep(tmp_path):
    graphiti = FakeGraphiti()
    client = _client(tmp_path, graphiti)
    client.graph.create(graph_id="g1", name="Docs")

    batch = client.batch.create(metadata={"graph_id": "g1"})
    added = client.batch.add(batch_id=batch.batch_id, items=[
        BatchAddItem(type="graph_episode", graph_id="g1", data=chunk, data_type="text")
        for chunk in ("chunk one", "chunk two")
    ])
    assert client.batch.get(batch_id=batch.batch_id).status == "draft"
    client.batch.process(batch_id=batch.batch_id)

    deadline = time.time() + 5
    while (summary := client.batch.get(batch_id=batch.batch_id)).status != "succeeded":
        assert time.time() < deadline, f"batch stuck in {summary.status}"
        time.sleep(0.01)

    assert summary.progress.percent_complete == 100
    assert summary.progress.succeeded_items == 2
    items = client.batch.list_items(batch_id=batch.batch_id, limit=100, cursor=None).items
    assert [(i.sequence_index, i.status, i.episode_uuid) for i in items] == [
        (0, "succeeded", added[0].episode_uuid),
        (1, "succeeded", added[1].episode_uuid),
    ]
    assert graphiti.bulk_calls[0][:2] == ("g1", ["chunk one", "chunk two"])


def test_graph_ontology_reaches_graphiti_and_survives_a_restart(tmp_path, monkeypatch):
    client = _client(tmp_path, FakeGraphiti())
    monkeypatch.setattr(graph_builder_module, "get_zep_client", lambda _key: client)
    builder = GraphBuilderService(api_key="unused")
    graph_id = builder.create_graph("Padel")
    builder.set_ontology(graph_id, {
        "entity_types": [
            {"name": "Company", "description": "A business.",
             "attributes": [{"name": "industry", "description": "Its industry"}]},
            {"name": "Club", "description": "A sports club.", "attributes": []},
        ],
        "edge_types": [
            {"name": "SPONSORS", "description": "Funds a club.", "attributes": [],
             "source_targets": [{"source": "Company", "target": "Club"}]},
        ],
    })

    restarted = FakeGraphiti()
    client = _client(tmp_path, restarted)
    _wait_processed(client, client.graph.add(graph_id=graph_id, type="text", data="x").uuid_)

    ontology = restarted.bulk_calls[0][2]
    company = ontology["entity_types"]["Company"]
    assert company.__doc__ == "A business."
    assert company.model_fields["industry"].description == "Its industry"
    assert set(ontology["entity_types"]) == {"Company", "Club"}
    assert set(ontology["edge_types"]) == {"SPONSORS"}
    assert ontology["edge_type_map"] == {("Company", "Club"): ["SPONSORS"]}


def test_graph_lifecycle_matches_zep_not_found_semantics(tmp_path, monkeypatch):
    cleared = []

    async def fake_clear_data(driver, group_ids):
        cleared.append(group_ids)

    monkeypatch.setattr(graphiti_zep, "clear_data", fake_clear_data)
    client = _client(tmp_path, FakeGraphiti())

    with pytest.raises(NotFoundError):
        client.graph.get("g1")
    client.graph.create(graph_id="g1", name="Padel", description="d")
    assert client.graph.get("g1").name == "Padel"

    client.graph.delete(graph_id="g1")

    assert cleared == [["g1"]]
    with pytest.raises(NotFoundError):
        client.graph.get("g1")


def test_nodes_come_back_zep_shaped_through_the_paging_helper(tmp_path, monkeypatch):
    node = EntityNode(
        name="Sara Khan", group_id="g1", labels=["Entity", "Person"],
        summary="Club manager", attributes={"role": "manager"}, name_embedding=[0.1],
    )

    async def fake_get_by_group_ids(driver, group_ids):
        assert group_ids == ["g1"]
        return [node]

    monkeypatch.setattr(graphiti_zep.EntityNode, "get_by_group_ids", fake_get_by_group_ids)
    client = _client(tmp_path, FakeGraphiti())

    [fetched] = fetch_all_nodes(client, "g1")

    assert fetched.uuid_ == node.uuid
    assert (fetched.name, fetched.labels, fetched.summary, fetched.attributes) == (
        "Sara Khan", ["Entity", "Person"], "Club manager", {"role": "manager"}
    )
    assert not hasattr(fetched, "name_embedding")


def test_search_uses_the_requested_scope_and_limit_without_llm_reranking(tmp_path):
    graphiti = FakeGraphiti()
    client = _client(tmp_path, graphiti)

    result = client.graph.search(
        graph_id="g1", query="who sponsors padel", limit=7, scope="nodes", reranker="rrf"
    )
    client.graph.search(
        graph_id="g1", query="sponsorship facts", limit=5, scope="edges",
        reranker="cross_encoder",
    )

    (query, node_config, groups), (_, edge_config, _) = graphiti.searches
    assert (query, groups, node_config.limit) == ("who sponsors padel", ["g1"], 7)
    assert node_config.node_config is not None and node_config.edge_config is None
    assert edge_config.edge_config.reranker == EdgeReranker.rrf
    assert result.nodes[0].name == "Ardila Sports"
    assert result.nodes[0].uuid_


def test_a_graph_without_edges_lists_none_instead_of_failing(tmp_path, monkeypatch):
    from graphiti_core.errors import GroupsEdgesNotFoundError

    async def no_edges(driver, group_ids):
        raise GroupsEdgesNotFoundError(group_ids)

    monkeypatch.setattr(graphiti_zep.EntityEdge, "get_by_group_ids", no_edges)
    client = _client(tmp_path, FakeGraphiti())

    assert fetch_all_edges(client, "g1") == []


def test_gemini_rate_limits_wait_and_retry_instead_of_failing_the_batch(monkeypatch):
    from graphiti_core.llm_client.errors import RateLimitError
    from graphiti_core.llm_client.gemini_client import GeminiClient

    calls, waits = [], []

    async def quota_then_ok(self, *args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise RateLimitError()
        return {"edges": []}

    async def no_wait(seconds):
        waits.append(seconds)

    monkeypatch.setattr(GeminiClient, "generate_response", quota_then_ok)
    monkeypatch.setattr(graphiti_zep.asyncio, "sleep", no_wait)
    client = graphiti_zep.PatientGeminiClient(graphiti_zep.LLMConfig(api_key="k"))

    assert asyncio.run(client.generate_response([Message(role="user", content="p")])) == {"edges": []}
    assert len(calls) == 2 and len(waits) == 1


def test_a_looping_or_overloaded_gemini_answer_gets_a_fresh_retry(monkeypatch):
    from graphiti_core.llm_client.gemini_client import GeminiClient

    seen = []

    async def loop_then_ok(self, messages, *args, **kwargs):
        seen.append([m.content for m in messages])
        messages[0].content += " [language note]"  # GeminiClient edits what it is given
        messages.append(Message(role="user", content="previous attempt was invalid"))
        if len(seen) == 1:
            raise Exception()  # what GeminiClient raises once its own two tries loop
        return {"summary": "ok"}

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(GeminiClient, "generate_response", loop_then_ok)
    monkeypatch.setattr(graphiti_zep.asyncio, "sleep", no_wait)
    client = graphiti_zep.PatientGeminiClient(graphiti_zep.LLMConfig(api_key="k"))

    assert asyncio.run(client.generate_response([Message(role="user", content="extract")])) == {
        "summary": "ok"
    }
    assert seen == [["extract"], ["extract"]]


def test_a_gemini_failure_that_persists_still_fails_after_a_few_fresh_tries(monkeypatch):
    from graphiti_core.llm_client.gemini_client import GeminiClient

    attempts = []

    async def always_loops(self, messages, *args, **kwargs):
        attempts.append(1)
        raise Exception()

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(GeminiClient, "generate_response", always_loops)
    monkeypatch.setattr(graphiti_zep.asyncio, "sleep", no_wait)
    client = graphiti_zep.PatientGeminiClient(graphiti_zep.LLMConfig(api_key="k"))

    with pytest.raises(Exception):
        asyncio.run(client.generate_response([Message(role="user", content="extract")]))
    assert len(attempts) == client.FAILURE_ATTEMPTS


def test_a_network_blip_while_embedding_waits_and_retries(monkeypatch):
    import httpx
    from graphiti_core.embedder.gemini import GeminiEmbedder

    calls, waits = [], []

    async def dns_blip_then_ok(self, input_data_list):
        calls.append(input_data_list)
        if len(calls) == 1:
            raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
        return [[0.1, 0.2]]

    async def record_wait(seconds):
        waits.append(seconds)

    monkeypatch.setattr(GeminiEmbedder, "create_batch", dns_blip_then_ok)
    monkeypatch.setattr(graphiti_zep.asyncio, "sleep", record_wait)
    embedder = graphiti_zep.PatientGeminiEmbedder(graphiti_zep.GeminiEmbedderConfig(api_key="k"))

    assert asyncio.run(embedder.create_batch(["Agent posted: hello"])) == [[0.1, 0.2]]
    assert len(calls) == 2 and len(waits) == 1


def test_network_and_overload_failures_get_minutes_of_patience_not_three_quick_tries(monkeypatch):
    import httpx
    from graphiti_core.llm_client.gemini_client import GeminiClient

    attempts, waits = [], []

    async def offline_for_a_while(self, messages, *args, **kwargs):
        attempts.append(1)
        if len(attempts) <= 4:
            try:  # GeminiClient wraps the transport error in a bare Exception
                raise httpx.ConnectError("[Errno 8] nodename nor servname provided")
            except httpx.ConnectError as error:
                raise Exception() from error
        return {"ok": True}

    async def record_wait(seconds):
        waits.append(seconds)

    monkeypatch.setattr(GeminiClient, "generate_response", offline_for_a_while)
    monkeypatch.setattr(graphiti_zep.asyncio, "sleep", record_wait)
    client = graphiti_zep.PatientGeminiClient(graphiti_zep.LLMConfig(api_key="k"))

    assert asyncio.run(client.generate_response([Message(role="user", content="extract")])) == {"ok": True}
    assert len(attempts) == 5 and sum(waits) >= 150

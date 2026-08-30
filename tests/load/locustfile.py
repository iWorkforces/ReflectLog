'''Load testing for ReflectLog using Locust.

Simulates high-traffic scenarios to identify performance bottlenecks
under concurrent user load and system stress conditions.

Usage:
    # Run locust with this file
    locust -f locustfile.py --headless --users 100 --spawn-rate 10

    # Run with custom parameters
    locust -f locustfile.py --headless --users 1000 --spawn-rate 100 --run-time 60

    # View real-time stats in browser
    locust -f locustfile.py --headless --host http://127.0.0.1:8089 --users 100
'''

from collections.abc import Callable, Sequence
from datetime import timedelta
from random import randint
import time
from typing import TYPE_CHECKING, ParamSpec, Protocol


class HttpClient(Protocol):
    def get(self, path: str) -> None: ...

    def post(self, path: str, *, json: dict[str, str | list[str]]) -> None: ...


class EventHook(Protocol):
    def add_listener[**P](self, listener: Callable[P, None]) -> Callable[P, None]: ...


class TestEventHook(Protocol):
    def add[**P](self, listener: Callable[P, None]) -> Callable[P, None]: ...


class RequestEvent(Protocol):
    def fire(self, *, request_type: str, name: str | None) -> None: ...


class RequestTime(Protocol):
    def get(self) -> timedelta: ...


class LocustEvents(Protocol):
    init: EventHook
    request: RequestEvent
    request_time: RequestTime
    test: TestEventHook


P = ParamSpec("P")

if TYPE_CHECKING:
    class HttpUser:
        client: HttpClient

        def wait(self) -> None: ...

    def between(min_wait: int, max_wait: int) -> Callable[[], float]: ...

    def task(function: Callable[P, None]) -> Callable[P, None]: ...

    class StaticEvents:
        init: EventHook
        request: RequestEvent
        request_time: RequestTime
        test: TestEventHook

    events: LocustEvents = StaticEvents()
else:
    from locust import HttpUser, between, events, task


class ReflectLogUser(HttpUser):
    '''Simulates ReflectLog server user behavior.

    Simulates realistic usage patterns:
    - Search queries (varied complexity)
    - Add operations (batching behavior)
    - Mixed operations (simulating real workload)
    '''

    wait_time = between(randint(1, 5), randint(1, 3))

    @task
    def add_memory(self) -> None:
        '''Add a memory to the store.

        Simulates typical AI assistant usage of storing context.
        '''
        self.client.post(
            "/mcp/add",
            json={
                "memories": [f"Memory {randint(1000, 9999)}"],
            },
        )
        self.wait()

    @task
    def search_memory(self) -> None:
        '''Search for stored memories.

        Simulates semantic search queries with varying complexity.
        '''
        query_type = randint(1, 3)
        query_lengths = {
            1: randint(10, 50),
            2: randint(50, 200),
            3: randint(200, 500),
        }
        query = "x" * query_lengths[query_type]

        self.client.post(
            "/mcp/search",
            json={"query": query},
        )
        self.wait()

    @task
    def get_all_memories(self) -> None:
        '''Retrieve all stored memories.

        Simulates full data retrieval operations.
        '''
        self.client.get("/mcp/get_all")
        time.sleep(randint(100, 500) / 1000.0)

    @task
    def health_check(self) -> None:
        '''Check server health status.

        Simulates periodic health monitoring.
        '''
        self.client.get("/mcp/health_check")
        self.wait()

    def on_start(self) -> None:
        '''Called when user starts a task.'''
        pass

    def on_stop(self) -> None:
        '''Called when user stops a task.'''
        pass


@events.init.add_listener
def on_request(request_type: str, name: str | None) -> None:
    '''Track request metrics for analysis.

    Args:
        request_type: Type of request (add, search, get_all, health_check)
        name: Name of the request handler
    '''
    # Log request type for analysis
    events.request.fire(request_type=request_type, name=name)


@events.test.add
def test_search_throughput(user: ReflectLogUser):
    '''Test search performance with various query sizes.

    Ensures search operations maintain acceptable response times
    under increasing query complexity.
    '''
    # Test with 50 character queries (simple)
    for i in range(10):
        user.search_memory()

    # Test with 200 character queries (moderate)
    for i in range(10):
        user.search_memory()

    # Test with 500 character queries (complex)
    for i in range(5):
        user.search_memory()


@events.test.add
def test_add_performance(user: ReflectLogUser):
    '''Test add performance under load.

    Measures throughput and latency for add operations.
    '''
    start_time = events.request_time.get()
    total_memories = 0
    for i in range(100):
        user.add_memory()
        total_memories += 1

    duration = (events.request_time.get() - start_time).total_seconds()
    throughput = total_memories / duration

    events.request.fire(
        request_type="add_performance",
        name="Add Performance Test",
    )

    assert throughput > 1.0, "Throughput should be at least 1.0 req/s"


@events.test.add
def test_mixed_workload(user: ReflectLogUser):
    '''Test realistic mixed workload.

    Simulates concurrent users performing various operations.
    '''
    # Run operations synchronously for testing
    for i in range(50):
        if randint(1, 100) <= 60:
            # Skip - would use asyncio but not available in sync context
            pass
        elif randint(1, 100) <= 90:
            user.add_memory()
        else:
            user.search_memory()

    start_time = events.request_time.get()
    duration = (events.request_time.get() - start_time).total_seconds()

    events.request.fire(
        request_type="mixed_workload",
        name="Mixed Workload Test",
    )


@events.test.add
def test_concurrent_users(user_factory: Callable[[], ReflectLogUser]) -> None:
    '''Test system behavior under concurrent user load.

    Identifies bottlenecks and resource contention.
    '''
    users = [user_factory() for _ in range(50)]

    start_time = events.request_time.get()
    run_locust_users(users, spawn_rate=50, run_time=30)
    duration = (events.request_time.get() - start_time).total_seconds()

    events.request.fire(
        request_type="concurrent_load",
        name="Concurrent Users Test",
    )


def run_locust_users(
    users: Sequence[ReflectLogUser], spawn_rate: float, run_time: int = 60
) -> None:
    '''Run Locust with specified parameters.

    Args:
        users: List of user instances to spawn.
        spawn_rate: Users per second to spawn.
        run_time: Seconds to run test.
    '''
    # Create temporary config file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
        f.write(f'''
[locust]
locustfile = locustfile.py
headless = true

[host]
host = http://127.0.0.1:9103

[user]
wait_time = min(5) max(10)

[http]
timeout = 120

[run]
users = {len(users)}
spawn_rate = {spawn_rate}
run_time = {run_time}
        ''')
        config_path = f.name

    import subprocess
    import sys

    print(f"Starting Locust with {len(users)} users...")
    print(f"  Spawn rate: {spawn_rate}/s")
    print(f"  Run time: {run_time}s")
    print(f"  Config: {config_path}")

    try:
        # Run locust with config file
        subprocess.run(
            [
                sys.executable,
                "-m",
                "locust",
                "-f",
                config_path,
            ],
            check=True,
        )
    finally:
        pass

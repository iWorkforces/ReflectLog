# Load Tests

**Generated:** 2026-02-22
**Commit:** 6c2d6fa
**Branch:** develop

## OVERVIEW

Locust-based load tests for MCP server performance testing. Measures throughput and latency under load.

## STRUCTURE

```
tests/load/
└── locustfile.py           # Locust user scenarios, add/search/remove operations
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add load | `locustfile.py:AddUser` class |
| Search load | `locustfile.py:SearchUser` class |
| Mixed workload | `locustfile.py:MixedUser` class |

## KEY PATTERNS

### Locust Task Definition
```python
class AddUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def add_memories(self):
        self.client.post("/add", json={
            "messages": [f"load test message {random.randint(1, 1000)}"]
        })
```

### Running Load Tests
```bash
# With Locust
locust -f tests/load/locustfile.py --host http://localhost:9103

# Headless
locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 60s
```

## ANTI-PATTERNS

- Never run against production
- Never skip wait_time (will DDOS server)

## NOTES

- **Requires running server**: Start with `./start-reflectlog-mcp-server.sh`
- **Metrics**: Use Locust web UI at http://localhost:8089
- **Not CI**: Manual performance testing only

import json


def load_names(raw: str) -> list[str]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Expected a list")
    return [str(item) for item in data]


for name in load_names('["Ada", "Grace"]'):
    print(name)

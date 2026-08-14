import json
import unittest

import pymap


SOURCE = '''import json

def parse(raw):
    if raw:
        return json.loads(raw)
    return None

for item in ["1", "2"]:
    print(parse(item))
'''


class FlowTests(unittest.TestCase):
    def test_analysis_has_catalog_and_execution_tree(self):
        result = pymap.analyze_python_source(SOURCE, "sample.py")
        self.assertEqual(result["catalog"]["functions"][0]["name"], "parse")
        kinds = [node["kind"] for node in result["flow"]["children"]]
        self.assertEqual(kinds, ["import", "function", "loop"])

    def test_function_body_is_deferred(self):
        result = pymap.analyze_python_source(SOURCE, "sample.py")
        body = result["flow"]["children"][1]["children"][0]
        self.assertEqual(body["execution"], "deferred")
        self.assertEqual(body["children"][0]["kind"], "condition")

    def test_json_output(self):
        encoded = pymap.map_analysis_to_json(pymap.analyze_python_source("x = 1"))
        self.assertEqual(json.loads(encoded)["flow"]["children"][0]["kind"], "assignment")


if __name__ == "__main__":
    unittest.main()

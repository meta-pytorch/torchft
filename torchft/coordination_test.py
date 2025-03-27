import inspect
from unittest import TestCase

from torchft.coordination import (
    LighthouseClient,
    LighthouseServer,
    ManagerClient,
    ManagerServer,
    Quorum,
    QuorumMember,
)


class TestCoordination(TestCase):
    def test_coordination_docs(self) -> None:
        classes = [
            ManagerClient,
            ManagerServer,
            LighthouseServer,
            LighthouseClient,
            Quorum,
            QuorumMember,
        ]
        for cls in classes:
            self.assertIn("Args:", str(cls.__doc__), cls)
            for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
                if name.startswith("_"):
                    continue
                self.assertIn("Args:", str(cls.__doc__), cls)

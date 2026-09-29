"""Allen CCF structure ontology: resolve regions to the full set of label IDs they cover."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import requests

STRUCTURE_GRAPH_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "exaspim-axon"


@dataclass(frozen=True)
class Structure:
    """One node of the CCF structure tree."""

    id: int
    acronym: str
    name: str
    color_hex: str
    parent_id: int | None
    children: tuple[int, ...] = field(default=(), repr=False)


class CcfOntology:
    """Lookup over the Allen Mouse Brain Atlas structure graph (graph id 1).

    Parameters
    ----------
    structures : iterable of Structure
        All structures in the tree.
    """

    def __init__(self, structures: Iterable[Structure]) -> None:
        self._by_id = {s.id: s for s in structures}
        self._by_acronym = {s.acronym.lower(): s for s in self._by_id.values()}

    @classmethod
    def load(cls, cache_dir: Path = DEFAULT_CACHE) -> CcfOntology:
        """Load the structure graph, downloading it once into ``cache_dir``."""
        path = cache_dir / "structure_graph_1.json"
        if not path.exists():
            response = requests.get(STRUCTURE_GRAPH_URL, timeout=60)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text)
        root = json.loads(path.read_text())["msg"][0]
        return cls(_flatten(root, parent_id=None))

    def __getitem__(self, key: int | str) -> Structure:
        """Look up a structure by numeric ID or (case-insensitive) acronym."""
        if isinstance(key, str) and not key.isdigit():
            return self._by_acronym[key.lower()]
        return self._by_id[int(key)]

    def descendants(self, key: int | str) -> set[int]:
        """IDs of a structure and every structure beneath it."""
        stack, out = [self[key].id], set()
        while stack:
            sid = stack.pop()
            out.add(sid)
            stack.extend(self._by_id[sid].children)
        return out

    def resolve(self, keys: Iterable[int | str]) -> set[int]:
        """Union of :meth:`descendants` over several regions."""
        return set().union(*(self.descendants(k) for k in keys))


def _flatten(node: dict, parent_id: int | None) -> Iterator[Structure]:
    yield Structure(
        id=node["id"],
        acronym=node["acronym"],
        name=node["name"],
        color_hex=node["color_hex_triplet"],
        parent_id=parent_id,
        children=tuple(child["id"] for child in node["children"]),
    )
    for child in node["children"]:
        yield from _flatten(child, parent_id=node["id"])

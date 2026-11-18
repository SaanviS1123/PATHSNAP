
import numpy as np
from scipy.spatial import cKDTree


class RoadCandidateIndex:
    """Spatial index for directed road segments."""

    def __init__(self, nodes, arcs, projection):
        self.arcs = np.asarray(arcs, dtype=int)

        xy = projection.transform(nodes)

        self.starts = xy[self.arcs[:, 0]]
        self.ends = xy[self.arcs[:, 1]]

        self.vectors = self.ends - self.starts

        self.lengths_sq = np.sum(
            self.vectors ** 2,
            axis=1
        )

        self.midpoints = (
            self.starts + self.ends
        ) / 2

        self.tree = cKDTree(self.midpoints)

    def query(self, point_xy, k=10, search_k=500):
        """Return nearest arc IDs and exact segment distances."""
        search_k = min(
            search_k,
            len(self.arcs)
        )

        _, ids = self.tree.query(
            point_xy,
            k=search_k
        )

        ids = np.atleast_1d(ids).astype(int)

        starts = self.starts[ids]
        vectors = self.vectors[ids]

        fractions = np.clip(
            np.sum(
                (point_xy - starts) * vectors,
                axis=1
            ) / np.maximum(
                self.lengths_sq[ids],
                1e-12
            ),
            0,
            1
        )

        projections = (
            starts + fractions[:, None] * vectors
        )

        distances = np.linalg.norm(
            projections - point_xy,
            axis=1
        )

        order = np.argsort(distances)[:k]

        return ids[order], distances[order]

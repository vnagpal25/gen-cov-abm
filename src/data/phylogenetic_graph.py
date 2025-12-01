from typing import Any, List, Set, Optional
from .genome_node import CovidGenomeSequence


class PhylogeneticGraph:
    """Graph-based structure to find smallest subtree containing specific nodes."""

    def __init__(self, root: CovidGenomeSequence):
        self.root = root
        self.parent_map = {}  # Maps node name to parent node
        self._build_parent_map(root, None)

    def _build_parent_map(
        self, node: CovidGenomeSequence, parent: Optional[CovidGenomeSequence]
    ):
        """Build a mapping from each node to its parent."""
        if parent is not None:
            self.parent_map[node.name] = parent
        for child in node.children:
            self._build_parent_map(child, node)

    def _get_path_to_root(self, node: CovidGenomeSequence) -> List[CovidGenomeSequence]:
        """Get the path from a node to the root."""
        path = [node]
        current = node
        while current.name in self.parent_map:
            current = self.parent_map[current.name]
            path.append(current)
        return path

    def find_lca(
        self, nodes: List[CovidGenomeSequence]
    ) -> Optional[CovidGenomeSequence]:
        """Find the Lowest Common Ancestor of a list of nodes."""
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]

        ancestor_sets = [set(self._get_path_to_root(node)) for node in nodes]
        common_ancestors = set.intersection(*ancestor_sets)
        if not common_ancestors:
            return None

        # Find the deepest common ancestor (lowest in path)
        for ancestor in self._get_path_to_root(nodes[0]):
            if ancestor in common_ancestors:
                return ancestor

    def get_smallest_subtree_nodes(
        self, nodes: List[CovidGenomeSequence]
    ) -> Set[CovidGenomeSequence]:
        """
        Return the exact set of nodes forming the smallest subtree
        containing all given nodes (from LCA downward).
        """
        if not nodes:
            return set()

        lca = self.find_lca(nodes)
        if lca is None:
            return set()

        # Build ancestor chains for all given nodes
        node_paths = {node: self._get_path_to_root(node) for node in nodes}

        # Collect nodes from each path, starting at LCA down to each tip
        subtree_nodes = set()
        for path in node_paths.values():
            if lca in path:
                lca_index = path.index(lca)
                subtree_nodes.update(path[: lca_index + 1])

        # Add descendants of LCA recursively to make a complete subtree
        self._collect_descendants(lca, subtree_nodes)
        return subtree_nodes

    def _collect_descendants(
        self, node: CovidGenomeSequence, result: Set[CovidGenomeSequence]
    ):
        """Recursively add all descendants of node to result."""
        for child in node.children:
            result.add(child)
            self._collect_descendants(child, result)

    def get_smallest_subtree_with_division(
        self, div: str
    ) -> Optional[CovidGenomeSequence]:
        """Return the root (LCA) of the smallest subtree containing all tips of a given division."""
        tips = self.find_tips(div)
        return self.find_lca(tips) if tips else None

    def find_tips(self, div: str) -> List[CovidGenomeSequence]:
        """Find all tip nodes with a given division."""
        tips = []
        self._collect_tips(self.root, div, tips)
        return tips

    def _collect_tips(self, node: CovidGenomeSequence, div: str, result: List):
        """Recursively collect tip nodes."""
        if node.is_collected_tip() and node.division == div:
            result.append(node)
        for child in node.children:
            self._collect_tips(child, div, result)

    def find_tips_by_attribute(
        self, attribute: str, value: Any
    ) -> List[CovidGenomeSequence]:
        """
        Find all tip nodes where a given attribute matches a specified value.

        Args:
            attribute: The attribute name to filter by (e.g., 'region', 'division', 'country', 'clade')
            value: The value to match for the specified attribute

        Returns:
            List of GenomeSequence tip nodes matching the criteria
        """
        tips = []
        self._collect_tips_by_attribute(self.root, attribute, value, tips)
        return tips

    def _collect_tips_by_attribute(
        self, node: CovidGenomeSequence, attribute: str, value: Any, result: List
    ):
        """Recursively collect tip nodes matching attribute criteria."""
        if node.is_collected_tip() and hasattr(node, attribute):
            if getattr(node, attribute) == value:
                result.append(node)
        for child in node.children:
            self._collect_tips_by_attribute(child, attribute, value, result)

    def get_smallest_subtree_with_attribute(
        self, attribute: str, value: Any
    ) -> Optional[CovidGenomeSequence]:
        """
        Return the root (LCA) of the smallest subtree containing all tips
        matching a given attribute-value pair.

        Args:
            attribute: The attribute name to filter by (e.g., 'region', 'division', 'country')
            value: The value to match for the specified attribute

        Returns:
            The LCA node of all matching tips, or None if no tips match
        """
        tips = self.find_tips_by_attribute(attribute, value)
        return self.find_lca(tips) if tips else None

    def find_tips_by_multiple_attributes(
        self, **attribute_filters
    ) -> List[CovidGenomeSequence]:
        """
        Find all tip nodes matching multiple attribute criteria.

        Args:
            **attribute_filters: Keyword arguments where keys are attribute names
                                and values are the desired values to match

        Returns:
            List of GenomeSequence tip nodes matching all specified criteria

        Example:
            find_tips_by_multiple_attributes(region='North America', clade='3C.2a1b')
        """
        tips = []
        self._collect_tips_by_multiple_attributes(self.root, attribute_filters, tips)
        return tips

    def _collect_tips_by_multiple_attributes(
        self, node: CovidGenomeSequence, attribute_filters: dict, result: List
    ):
        """Recursively collect tip nodes matching all attribute criteria."""
        if node.is_collected_tip():
            # Check if all attribute filters match
            match = all(
                hasattr(node, attr) and getattr(node, attr) == value
                for attr, value in attribute_filters.items()
            )
            if match:
                result.append(node)
        for child in node.children:
            self._collect_tips_by_multiple_attributes(child, attribute_filters, result)

    def get_ancestor_union(
        self, nodes: List[CovidGenomeSequence]
    ) -> Set[CovidGenomeSequence]:
        """
        Return the union of all ancestor chains for the given nodes.
        Includes all nodes that lie on any path from the root to any of the input nodes.
        The only tips in this set will be the given nodes themselves.
        """
        ancestor_union = set()
        for node in nodes:
            path = self._get_path_to_root(node)
            ancestor_union.update(path)
        return ancestor_union

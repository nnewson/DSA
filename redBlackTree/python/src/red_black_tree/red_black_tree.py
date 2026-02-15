from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, Protocol, TypeVar


class SupportsLT(Protocol):
    """Protocol for objects that support less-than comparison operations.

    This protocol defines the interface for any type that implements
    the less-than (__lt__) operator, enabling comparison-based operations
    such as sorting and ordering.
    """

    def __lt__(self, other: object) -> bool: ...


K = TypeVar("K", bound=SupportsLT)
V = TypeVar("V")


class Colour(Enum):
    """Enumeration for Red-Black Tree node colors.

    Attributes:
        RED: Represents a red node in the Red-Black Tree.
        BLACK: Represents a black node in the Red-Black Tree.
    """

    RED = 0
    BLACK = 1


# Using slots=true to save the overhead of the __dict__ for each node,
# since we know exactly which attributes we need.
@dataclass(slots=True)
class _Node(Generic[K, V]):
    key: K
    value: V
    colour: Colour = Colour.RED
    left: Optional[_Node[K, V]] = None
    right: Optional[_Node[K, V]] = None
    parent: Optional[_Node[K, V]] = None


class RedBlackTree(Generic[K, V]):
    def __init__(self):
        # Sentinel node for leaves to simplify the logic for the delete operation and to
        # avoid having to check for None in various places.
        self._nil = _Node(None, None, colour=Colour.BLACK)
        self._nil.left = self._nil.right = self._nil.parent = self._nil

        self.root: _Node[K, V] = self._nil
        self.size: int = 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        """In-order traversal of the Red-Black Tree.

        This method allows iteration over the key-value pairs in the tree in sorted
        order based on the keys. It uses a generator to yield each key-value pair
        as a tuple (key, value) during the traversal.

        Yields:
            A tuple containing the key and value of each node in the tree, in sorted order.
        """

        def _in_order_traversal(node: _Node[K, V]):
            if node is not self._nil:
                yield from _in_order_traversal(node.left)
                yield (node.key, node.value)
                yield from _in_order_traversal(node.right)

        yield from _in_order_traversal(self.root)

    def insert(self, key: K, value: V) -> None:
        """Inserts a key-value pair into the Red-Black Tree.

        This method adds a new node with the specified key and value to the tree,
        maintaining the properties of the Red-Black Tree to ensure balanced
        structure and efficient operations.

        Args:
            key: The key associated with the value to be inserted. Must support
                 less-than comparison.
            value: The value to be stored in the tree corresponding to the key.
        """
        parent_node: _Node[K, V]
        found: bool
        parent_node, found = self._find_node_or_parent(key)
        if found:
            # If the key already exists, update the value and return
            parent_node.value = value
            return

        # Adding a new node to the tree
        self.size += 1
        new_node = _Node(key, value, left=self._nil, right=self._nil, parent=self._nil)

        if parent_node is self._nil:
            # The tree was empty, so the new node becomes the root.
            self.root = new_node
        elif key < parent_node.key:
            parent_node.left = new_node
            new_node.parent = parent_node
        else:
            parent_node.right = new_node
            new_node.parent = parent_node

        self._insert_fixup(new_node)

    def find(self, key: K) -> Optional[V]:
        """Finds the value associated with a given key in the Red-Black Tree.

        This method searches for a node with the specified key and returns its
        associated value if found. If the key is not present in the tree, it
        returns None.

        Args:
            key: The key to search for in the tree. Must support less-than
                 comparison.

        Returns:
            The value associated with the specified key if found; otherwise, None.
        """
        node: _Node[K, V]
        found: bool
        node, found = self._find_node_or_parent(key)
        return node.value if found else None

    def delete(self, key: K) -> bool:
        """Deletes a node with the specified key from the Red-Black Tree.

        This method removes the node with the given key from the tree while
        maintaining the properties of the Red-Black Tree to ensure balanced
        structure and efficient operations.

        Args:
            key: The key of the node to be deleted. Must support less-than
                 comparison.

        Returns:
            True if the node was found and deleted, False otherwise.
        """
        node_to_delete: _Node[K, V]
        found: bool
        node_to_delete, found = self._find_node_or_parent(key)
        if not found:
            return False

        node: _Node[K, V] = node_to_delete
        node_original_colour = node.colour

        fixup_node: _Node[K, V]

        if node_to_delete.left is self._nil:
            fixup_node = node_to_delete.right
            self._transplant(node_to_delete, node_to_delete.right)
        elif node_to_delete.right is self._nil:
            fixup_node = node_to_delete.left
            self._transplant(node_to_delete, node_to_delete.left)
        else:
            successor = self._minimum(node_to_delete.right)
            node_original_colour = successor.colour
            fixup_node = successor.right

            if successor.parent == node_to_delete:
                fixup_node.parent = successor
            else:
                self._transplant(successor, successor.right)
                successor.right = node_to_delete.right
                successor.right.parent = successor

            self._transplant(node_to_delete, successor)
            successor.left = node_to_delete.left
            successor.left.parent = successor
            successor.colour = node_to_delete.colour

        if node_original_colour == Colour.BLACK:
            self._delete_fixup(fixup_node)

        self.size -= 1
        return True

    def _minimum(self, node: _Node[K, V]) -> _Node[K, V]:
        """Finds the node with the minimum key in the subtree rooted at the given node.

        This method traverses the left children of the subtree until it reaches a
        leaf node (nil), which is the node with the smallest key in that subtree.

        Args:
            node: The root of the subtree to search for the minimum key.

        Returns:
            The node with the minimum key in the specified subtree.
        """
        current_node = node
        while current_node.left is not self._nil:
            current_node = current_node.left
        return current_node

    def _transplant(self, from_node: _Node[K, V], to_node: _Node[K, V]) -> None:
        """Replaces one subtree as a child of its parent with another subtree.

        This method is used during the delete operation to replace the subtree
        rooted at node u with the subtree rooted at node v. It updates the parent
        pointers accordingly to maintain the tree structure.

        Args:
            u: The node to be replaced (the subtree rooted at this node will be removed).
            v: The node to replace u (the subtree rooted at this node will take u's place).

        Returns:
            None
        """
        if from_node.parent is self._nil:
            self.root = to_node
        elif from_node == from_node.parent.left:
            from_node.parent.left = to_node
        else:
            from_node.parent.right = to_node

        to_node.parent = from_node.parent

    def _find_node_or_parent(self, key: K) -> tuple[_Node[K, V], bool]:
        """
        Find a node with the given key or locate its parent node.

        Traverses the tree starting from the root to search for a node matching the given key.
        If the key is found, returns the node and True. If the key is not found, returns the
        parent node where a new node with this key should be inserted and False.

        Args:
            key: The key to search for in the tree.

        Returns:
            A tuple containing:
            - _Node[K, V]: The node with the matching key if found, or the parent node
                           where a new node should be inserted if not found.
            - bool: True if the key was found, False otherwise.
        """
        current_node = self.root
        parent_node: _Node[K, V] = self._nil

        while current_node is not self._nil:
            parent_node = current_node
            if key < current_node.key:
                current_node = current_node.left
            elif current_node.key < key:
                current_node = current_node.right
            else:
                return current_node, True  # Key found

        return (
            parent_node,
            False,
        )  # Key not found, return parent for potential insertion

    def _delete_fixup(self, node: _Node[K, V]) -> None:
        """
        Restore Red-Black Tree properties after deletion.

        Fixes violations of Red-Black Tree properties that may occur after deleting
        a node. Uses the standard CLRS algorithm by examining the sibling and its
        children, then performing rotations and recoloring as necessary.

        The method handles several cases based on the color of the sibling and its
        children:
        - Case 1: Sibling is red - recolor sibling and parent, then rotate to move
                  the red sibling up the tree.
        - Case 2: Sibling is black with two black children - recolor sibling to red
                  and move up the tree to fix potential violations.
        - Case 3: Sibling is black with one red child (the far child) -
                  recolor sibling and parent, then rotate to fix the violation.
        - Case 4: Sibling is black with one red child (the near child) -
                  recolor sibling and its red child, then rotate to fix the violation.

        The process continues up the tree until the root is reached or no violations remain.
        Args:
            node: The node that may violate Red-Black Tree properties after deletion.

        Returns:
            None
        """
        while node is not self.root and (
            node is self._nil or node.colour == Colour.BLACK
        ):
            if node == node.parent.left:
                sibling = node.parent.right
                if sibling.colour == Colour.RED:
                    # Case 1: Sibling is red
                    sibling.colour = Colour.BLACK
                    node.parent.colour = Colour.RED
                    self._rotate_left(node.parent)
                    sibling = node.parent.right

                if (
                    sibling.left.colour == Colour.BLACK
                    and sibling.right.colour == Colour.BLACK
                ):
                    # Case 2: Sibling is black with two black children
                    sibling.colour = Colour.RED
                    node = node.parent
                else:
                    if sibling.right.colour == Colour.BLACK:
                        # Case 4: Sibling is black with one red child (the near child)
                        sibling.left.colour = Colour.BLACK
                        sibling.colour = Colour.RED
                        self._rotate_right(sibling)
                        sibling = node.parent.right

                    # Case 3: Sibling is black with one red child (the far child)
                    sibling.colour = node.parent.colour
                    node.parent.colour = Colour.BLACK
                    sibling.right.colour = Colour.BLACK
                    self._rotate_left(node.parent)
                    node = self.root
            else:
                sibling = node.parent.left
                if sibling.colour == Colour.RED:
                    # Case 1: Sibling is red
                    sibling.colour = Colour.BLACK
                    node.parent.colour = Colour.RED
                    self._rotate_right(node.parent)
                    sibling = node.parent.left

                if (
                    sibling.right.colour == Colour.BLACK
                    and sibling.left.colour == Colour.BLACK
                ):
                    # Case 2: Sibling is black with two black children
                    sibling.colour = Colour.RED
                    node = node.parent
                else:
                    if sibling.left.colour == Colour.BLACK:
                        # Case 4: Sibling is black with one red child (the near child)
                        sibling.right.colour = Colour.BLACK
                        sibling.colour = Colour.RED
                        self._rotate_left(sibling)
                        sibling = node.parent.left

                    # Case 3: Sibling is black with one red child (the far child)
                    sibling.colour = node.parent.colour
                    node.parent.colour = Colour.BLACK
                    sibling.left.colour = Colour.BLACK
                    self._rotate_right(node.parent)
                    node = self.root

        if node is not self._nil:
            node.colour = Colour.BLACK

    def _insert_fixup(self, node: _Node[K, V]) -> None:
        """
        Restore Red-Black Tree properties after insertion.

        Fixes violations of Red-Black Tree properties that may occur after inserting
        a new node. Uses the standard CLRS algorithm by examining the colors of the
        parent and uncle nodes, then performing rotations and recoloring as necessary.

        The method handles three main cases:
        - Case 1: Uncle is red - recolor parent, uncle, and grandparent
        - Case 2: Uncle is black and node is on opposite side of parent - rotate to align node
        - Case 3: Uncle is black and node is on same side as parent - rotate and recolor

        The process continues up the tree until the root is reached or no violations remain.

        Args:
            node: The newly inserted node that may violate Red-Black Tree properties.

        Returns:
            None
        """
        while node.parent is not self._nil and node.parent.colour == Colour.RED:
            # Check if the node's parent is a left child or a right child of the grandparent
            # to determine the uncle's position.
            if node.parent == node.parent.parent.left:
                # Uncle is the right child of the grandparent.
                uncle = node.parent.parent.right
                if uncle is not self._nil and uncle.colour == Colour.RED:
                    # Case 1: Parent red, Uncle is red (recoloring)
                    node.parent.colour = Colour.BLACK
                    uncle.colour = Colour.BLACK
                    node.parent.parent.colour = Colour.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Uncle is black and current node is a right child
                        # (left rotation needed)
                        node = node.parent
                        self._rotate_left(node)
                    # Case 3: Uncle is black and current node is a left child
                    # (right rotation needed)
                    node.parent.colour = Colour.BLACK
                    node.parent.parent.colour = Colour.RED
                    self._rotate_right(node.parent.parent)
            else:
                # Uncle is the left child of the grandparent.
                uncle = node.parent.parent.left
                if uncle is not self._nil and uncle.colour == Colour.RED:
                    # Case 1: Parent red, Uncle is red (recoloring)
                    node.parent.colour = Colour.BLACK
                    uncle.colour = Colour.BLACK
                    node.parent.parent.colour = Colour.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        # Case 2: Uncle is black and current node is a left child
                        # (right rotation needed)
                        node = node.parent
                        self._rotate_right(node)
                    # Case 3: Uncle is black and current node is a right child
                    # (left rotation needed)
                    node.parent.colour = Colour.BLACK
                    node.parent.parent.colour = Colour.RED
                    self._rotate_left(node.parent.parent)

        self.root.colour = Colour.BLACK

    def _rotate_left(self, node: _Node[K, V]) -> None:
        """
        Performs a left rotation on the given node in the red-black tree.

        A left rotation restructures the tree by moving the node's right child up to
        take the node's position, and moving the node down to be the left child of
        its former right child. The left subtree of the right child is transferred
        to become the right subtree of the original node.

        Before rotation:                    After rotation:

            node                             right_child
             / \\                              / \\
           A   right_child    -->         node     C
             /  \\                         / \\
            B    C                        A   B

        Args:
            node (_Node[K, V]): The node to rotate left. Must have a non-nil right child.

        Raises:
            AssertionError: If the node's right child is None or nil.

        Note:
            This operation modifies parent-child relationships but maintains the
            in-order traversal property of the tree. It is typically used during
            red-black tree rebalancing operations.
        """
        right_child = node.right
        # We expect the right child to exist since we're rotating left on this node.
        assert right_child is not None and right_child is not self._nil

        # Move the left subtree of the right child to be the right subtree of the
        # current node.
        node.right = right_child.left
        if right_child.left is not self._nil:
            right_child.left.parent = node

        # Update the parent of the right child to be the parent of the current node.
        right_child.parent = node.parent

        # Update the parent's child pointer to point to the right child instead of the current node.
        if node.parent is self._nil:
            self.root = right_child
        elif node == node.parent.left:
            node.parent.left = right_child
        else:
            node.parent.right = right_child

        # Move the current node to be the left child of the right child.
        right_child.left = node
        node.parent = right_child

    def _rotate_right(self, node: _Node[K, V]) -> None:
        """
        Performs a right rotation on the given node in the red-black tree.

        A right rotation restructures the tree by moving the node's left child up to
        take the node's position, and moving the node down to be the right child of
        its former left child. The right subtree of the left child is transferred
        to become the right subtree of the original node.

        Before rotation:                    After rotation:

                 node                         left_child
                 /  \\                          / \\                    
        left_child   C        -->              A    node
           /  \\                                     / \\ 
          A    B                                    B   C

        Args:
            node (_Node[K, V]): The node to rotate right. Must have a non-nil left child.

        Raises:
            AssertionError: If the node's left child is None or nil.

        Note:
            This operation modifies parent-child relationships but maintains the
            in-order traversal property of the tree. It is typically used during
            red-black tree rebalancing operations.
        """
        left_child = node.left
        # We expect the left child to exist since we're rotating right on this node.
        assert left_child is not None and left_child is not self._nil

        # Move the right subtree of the left child to be the left subtree of the
        # current node.
        node.left = left_child.right
        if left_child.right is not self._nil:
            left_child.right.parent = node

        # Update the parent of the left child to be the parent of the current node.
        left_child.parent = node.parent

        # Update the parent's child pointer to point to the left child instead of the current node.
        if node.parent is self._nil:
            self.root = left_child
        elif node == node.parent.right:
            node.parent.right = left_child
        else:
            node.parent.left = left_child

        # Move the current node to be the right child of the left child.
        left_child.right = node
        node.parent = left_child

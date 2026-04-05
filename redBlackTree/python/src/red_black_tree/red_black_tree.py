from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Protocol


class SupportsLT(Protocol):
    """Protocol for objects that support less-than comparison operations.

    This protocol defines the interface for any type that implements
    the less-than (__lt__) operator, enabling comparison-based operations
    such as sorting and ordering.
    """

    def __lt__(self, other: object) -> bool: ...


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
class _Node[K: SupportsLT, V]:
    key: K
    value: V
    colour: Colour = Colour.RED
    nil: InitVar[_Node[K, V] | None] = None
    left: _Node[K, V] = field(init=False)
    right: _Node[K, V] = field(init=False)
    parent: _Node[K, V] = field(init=False)

    def __post_init__(self, nil: _Node[K, V] | None):
        sentinel = nil if nil is not None else self
        self.left = sentinel
        self.right = sentinel
        self.parent = sentinel


class RedBlackTree[K: SupportsLT, V]:
    def __init__(self):
        # Sentinel node for leaves to simplify the logic for the delete operation and to
        # avoid having to check for None in various places.
        # The sentinel's left, right, and parent are self-referential via __post_init__.
        self._nil: _Node[K, V] = _Node(None, None, colour=Colour.BLACK)  # type: ignore
        self.root: _Node[K, V] = self._nil
        self.size: int = 0

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: K) -> bool:
        _, found = self._find_node_or_parent(key)
        return found

    def __getitem__(self, key: K) -> V:
        node, found = self._find_node_or_parent(key)
        if not found:
            raise KeyError(key)
        return node.value

    def __iter__(self):
        """In-order traversal of the Red-Black Tree.

        This method allows iteration over the key-value pairs in the tree in sorted
        order based on the keys. Uses an explicit stack to avoid recursion limits.

        Yields:
            A tuple containing the key and value of each node in the tree, in sorted order.
        """
        stack: list[_Node[K, V]] = []
        node = self.root
        while stack or node is not self._nil:
            while node is not self._nil:
                stack.append(node)
                node = node.left

            node = stack.pop()
            yield (node.key, node.value)

            node = node.right

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
        new_node = _Node(key, value, nil=self._nil)

        if parent_node is self._nil:
            # The tree was empty, so the new node becomes the root.
            self.root = new_node
        elif key < parent_node.key:
            parent_node.left = new_node
            new_node.parent = parent_node
        else:
            parent_node.right = new_node
            new_node.parent = parent_node

        # After inserting the new node, we need to fix any violations of the Red-Black Tree properties.
        self._insert_fixup(new_node)

    def find(self, key: K) -> V | None:
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

        node_original_colour = node_to_delete.colour

        fixup_node: _Node[K, V]

        # Handle the three cases for deletion:
        if node_to_delete.left is self._nil:
            # 1. Node to delete has no left child (or is a leaf)
            fixup_node = node_to_delete.right
            self._transplant(node_to_delete, node_to_delete.right)
        elif node_to_delete.right is self._nil:
            # 2. Node to delete has no right child
            fixup_node = node_to_delete.left
            self._transplant(node_to_delete, node_to_delete.left)
        else:
            # 3. Node to delete has two children (find successor and replace)
            successor = self._minimum(node_to_delete.right)
            node_original_colour = successor.colour
            fixup_node = successor.right

            if successor.parent == node_to_delete:
                # If the successor is the direct child of the node to delete, we can directly
                # transplant the successor to the node's position.
                fixup_node.parent = successor
            else:
                # Otherwise, we need to first replace the successor with its right child
                # and then replace the node to delete with the successor.
                self._transplant(successor, successor.right)
                successor.right = node_to_delete.right
                successor.right.parent = successor

            # Finally, replace the node to delete with the successor.
            self._transplant(node_to_delete, successor)
            successor.left = node_to_delete.left
            successor.left.parent = successor
            successor.colour = node_to_delete.colour

        # If the original colour of the node being deleted was black, we need to fix up the tree
        # to restore the Red-Black Tree properties.
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
        """
        Replace a subtree rooted at from_node with a subtree rooted at to_node.

        This is a helper method used during node deletion in a Red-Black tree.
        It updates the parent pointer of to_node and the appropriate child pointer
        of from_node's parent to point to to_node instead of from_node.

        Args:
            from_node: The node to be replaced.
            to_node: The node that will replace from_node.

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
            # Update the parent node before moving down the tree.
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

        Four cases are handled (with symmetric mirrors for left/right):
        - Case 1: Sibling is red — rotate parent to make sibling black,
                  converting to Cases 2, 3, or 4.
        - Case 2: Sibling is black with two black children — recolor sibling to red
                  and move up the tree to fix potential violations.
        - Case 3: Sibling is black, far child is black, near child is red —
                  rotate sibling to make the far child red, converting to Case 4.
        - Case 4: Sibling is black, far child is red — rotate parent and recolour
                  to absorb the extra black. This is the terminal case.

        The process continues up the tree until the root is reached or no violations remain.

        Args:
            node: The node that may violate Red-Black Tree properties after deletion.
        """
        while node is not self.root and node.colour == Colour.BLACK:
            is_left = node == node.parent.left
            sibling = node.parent.right if is_left else node.parent.left

            if sibling.colour == Colour.RED:
                # Case 1: sibling is red — rotate to get a black sibling
                sibling.colour = Colour.BLACK
                node.parent.colour = Colour.RED
                (self._rotate_left if is_left else self._rotate_right)(node.parent)
                sibling = node.parent.right if is_left else node.parent.left

            near_nephew = sibling.left if is_left else sibling.right
            far_nephew = sibling.right if is_left else sibling.left

            if near_nephew.colour == Colour.BLACK and far_nephew.colour == Colour.BLACK:
                # Case 2: both nephews black — pull black up to parent
                sibling.colour = Colour.RED
                node = node.parent
                continue

            if far_nephew.colour == Colour.BLACK:
                # Case 3: near nephew red, far nephew black — rotate sibling
                near_nephew.colour = Colour.BLACK
                sibling.colour = Colour.RED
                (self._rotate_right if is_left else self._rotate_left)(sibling)
                sibling = node.parent.right if is_left else node.parent.left
                far_nephew = sibling.right if is_left else sibling.left

            # Case 4: far nephew is red — absorb the extra black (terminal)
            sibling.colour = node.parent.colour
            node.parent.colour = Colour.BLACK
            far_nephew.colour = Colour.BLACK
            (self._rotate_left if is_left else self._rotate_right)(node.parent)
            node = self.root

        node.colour = Colour.BLACK

    def _insert_fixup(self, node: _Node[K, V]) -> None:
        """
        Restore Red-Black Tree properties after insertion.

        Fixes violations of Red-Black Tree properties that may occur after inserting
        a new node. Uses the standard CLRS algorithm by examining the colors of the
        parent and uncle nodes, then performing rotations and recoloring as necessary.

        Three cases are handled (with symmetric mirrors for left/right):
        - Case 1: Uncle is red — recolor parent, uncle, and grandparent;
                  move the violation up two levels and repeat.
        - Case 2: Uncle is black, node is an inner child (zig-zag) — rotate
                  node's parent to convert into Case 3.
        - Case 3: Uncle is black, node is an outer child (zig-zig) — recolour
                  and rotate grandparent. This is the terminal case.

        Args:
            node: The newly inserted node that may violate Red-Black Tree properties.
        """
        while node.parent is not self._nil and node.parent.colour == Colour.RED:
            grandparent = node.parent.parent
            parent_is_left = node.parent == grandparent.left
            uncle = grandparent.right if parent_is_left else grandparent.left

            if uncle is not self._nil and uncle.colour == Colour.RED:
                # Case 1: uncle is red — push blackness down from grandparent
                node.parent.colour = Colour.BLACK
                uncle.colour = Colour.BLACK
                grandparent.colour = Colour.RED
                node = grandparent
                continue

            # Uncle is black — determine if node is an inner child (zig-zag)
            node_is_inner = (
                (node == node.parent.right)
                if parent_is_left
                else (node == node.parent.left)
            )

            if node_is_inner:
                # Case 2: zig-zag — rotate to align as outer child, converting to Case 3
                node = node.parent
                (self._rotate_left if parent_is_left else self._rotate_right)(node)

            # Case 3: zig-zig — recolour and rotate grandparent
            node.parent.colour = Colour.BLACK
            grandparent.colour = Colour.RED
            (self._rotate_right if parent_is_left else self._rotate_left)(grandparent)

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

from red_black_tree import RedBlackTree
from red_black_tree.red_black_tree import Colour, _Node


# ──────────────────────────────────────────────
#  Helpers for Red-Black Tree invariant checking
# ──────────────────────────────────────────────


def collect_inorder(tree):
    """Collect (key, value) pairs via in-order traversal."""
    result = []

    def _walk(node):
        if node is tree._nil or node is None:
            return
        _walk(node.left)
        result.append((node.key, node.value))
        _walk(node.right)

    _walk(tree.root)
    return result


def compute_black_height(tree, node):
    """Return the black-height of a node, or raise if inconsistent."""
    if node is tree._nil or node is None:
        return 1  # NIL nodes count as black
    left_bh = compute_black_height(tree, node.left)
    right_bh = compute_black_height(tree, node.right)
    assert left_bh == right_bh, (
        f"Black-height mismatch at key={node.key}: "
        f"left={left_bh}, right={right_bh}"
    )
    return left_bh + (1 if node.colour == Colour.BLACK else 0)


def assert_no_red_red(tree, node=None):
    """Assert no red node has a red child."""
    if node is None:
        node = tree.root
    if node is tree._nil or node is None:
        return
    if node.colour == Colour.RED:
        if node.left is not tree._nil and node.left is not None:
            assert node.left.colour == Colour.BLACK, (
                f"Red-red violation: node {node.key} (red) "
                f"has red left child {node.left.key}"
            )
        if node.right is not tree._nil and node.right is not None:
            assert node.right.colour == Colour.BLACK, (
                f"Red-red violation: node {node.key} (red) "
                f"has red right child {node.right.key}"
            )
    assert_no_red_red(tree, node.left)
    assert_no_red_red(tree, node.right)


def assert_parent_pointers(tree, node=None, expected_parent=None):
    """Verify every node's parent pointer is correct."""
    if node is None:
        node = tree.root
        expected_parent = tree._nil
    if node is tree._nil or node is None:
        return

    expected_label = (
        expected_parent.key
        if expected_parent is not tree._nil and expected_parent is not None
        else "NIL"
    )
    actual_label = (
        node.parent.key
        if node.parent is not tree._nil and node.parent is not None
        else repr(node.parent)
    )
    assert node.parent is expected_parent, (
        f"Node {node.key}: parent should be {expected_label} but is {actual_label}"
    )
    assert_parent_pointers(tree, node.left, node)
    assert_parent_pointers(tree, node.right, node)


def assert_rb_properties(tree):
    """Assert all Red-Black Tree invariants hold."""
    if tree.root is tree._nil or tree.root is None:
        return  # Empty tree trivially satisfies all properties

    # Property 2: Root is black
    assert tree.root.colour == Colour.BLACK, "Root must be black"

    # Property 4: No red-red violations
    assert_no_red_red(tree)

    # Property 5: Consistent black-height
    compute_black_height(tree, tree.root)

    # BST ordering invariant
    pairs = collect_inorder(tree)
    keys = [k for k, _ in pairs]
    assert keys == sorted(keys), f"BST ordering violated: {keys}"

    # Parent pointer consistency
    assert_parent_pointers(tree)


# ──────────────────────────────────────────────
#  Colour Enum
# ──────────────────────────────────────────────


class TestColourEnum:
    def test_red_value(self):
        assert Colour.RED.value == 0

    def test_black_value(self):
        assert Colour.BLACK.value == 1

    def test_red_is_not_black(self):
        assert Colour.RED != Colour.BLACK

    def test_two_members_only(self):
        assert len(Colour) == 2


# ──────────────────────────────────────────────
#  _Node Dataclass
# ──────────────────────────────────────────────


class TestNode:
    def test_default_colour_is_red(self):
        node = _Node(key=1, value="a")
        assert node.colour == Colour.RED

    def test_default_children_are_none(self):
        node = _Node(key=1, value="a")
        assert node.left is None
        assert node.right is None

    def test_default_parent_is_none(self):
        node = _Node(key=1, value="a")
        assert node.parent is None

    def test_explicit_black_colour(self):
        node = _Node(key=1, value="a", colour=Colour.BLACK)
        assert node.colour == Colour.BLACK

    def test_stores_key_and_value(self):
        node = _Node(key=42, value="hello")
        assert node.key == 42
        assert node.value == "hello"

    def test_uses_slots(self):
        node = _Node(key=1, value="a")
        assert not hasattr(node, "__dict__")


# ──────────────────────────────────────────────
#  Empty Tree
# ──────────────────────────────────────────────


class TestEmptyTree:
    def test_initial_size_is_zero(self):
        tree = RedBlackTree()
        assert len(tree) == 0

    def test_size_attribute_is_zero(self):
        tree = RedBlackTree()
        assert tree.size == 0

    def test_root_is_nil_sentinel(self):
        tree = RedBlackTree()
        assert tree.root is tree._nil

    def test_find_returns_none(self):
        tree = RedBlackTree()
        assert tree.find(42) is None

    def test_delete_returns_false(self):
        tree = RedBlackTree()
        assert tree.delete(42) is False

    def test_nil_sentinel_is_black(self):
        tree = RedBlackTree()
        assert tree._nil.colour == Colour.BLACK

    def test_nil_sentinel_is_self_referential(self):
        tree = RedBlackTree()
        assert tree._nil.left is tree._nil
        assert tree._nil.right is tree._nil
        assert tree._nil.parent is tree._nil

    def test_find_multiple_missing_keys(self):
        tree = RedBlackTree()
        for k in [0, -1, 100, "abc"]:
            assert tree.find(k) is None

    def test_delete_multiple_missing_keys(self):
        tree = RedBlackTree()
        for k in [0, -1, 100]:
            assert tree.delete(k) is False


# ──────────────────────────────────────────────
#  Single Element Insert
# ──────────────────────────────────────────────


class TestSingleInsert:
    def test_size_is_one(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert len(tree) == 1

    def test_find_returns_value(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.find(10) == "ten"

    def test_root_is_black(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.root.colour == Colour.BLACK

    def test_root_key_matches(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.root.key == 10

    def test_root_children_are_nil(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.root.left is tree._nil
        assert tree.root.right is tree._nil

    def test_root_parent_is_nil(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.root.parent is tree._nil

    def test_find_missing_key_returns_none(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.find(999) is None

    def test_rb_properties(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert_rb_properties(tree)


# ──────────────────────────────────────────────
#  Duplicate Key (Value Update)
# ──────────────────────────────────────────────


class TestDuplicateInsert:
    def test_updates_value(self):
        tree = RedBlackTree()
        tree.insert(10, "old")
        tree.insert(10, "new")
        assert tree.find(10) == "new"

    def test_size_unchanged(self):
        tree = RedBlackTree()
        tree.insert(10, "old")
        tree.insert(10, "new")
        assert len(tree) == 1

    def test_multiple_updates(self):
        tree = RedBlackTree()
        tree.insert(5, "a")
        tree.insert(5, "b")
        tree.insert(5, "c")
        assert tree.find(5) == "c"
        assert len(tree) == 1

    def test_update_preserves_rb_properties(self):
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        tree.insert(5, "updated")
        assert_rb_properties(tree)
        assert tree.find(5) == "updated"


# ──────────────────────────────────────────────
#  Two Element Inserts
# ──────────────────────────────────────────────


class TestTwoInserts:
    def test_insert_smaller_then_larger(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.insert(5, "five")
        assert len(tree) == 2
        assert tree.find(5) == "five"
        assert tree.find(10) == "ten"
        assert_rb_properties(tree)

    def test_insert_larger_then_smaller(self):
        tree = RedBlackTree()
        tree.insert(5, "five")
        tree.insert(10, "ten")
        assert len(tree) == 2
        assert tree.find(5) == "five"
        assert tree.find(10) == "ten"
        assert_rb_properties(tree)

    def test_child_is_red(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.insert(5, "five")
        # Root should be black, child should be red
        assert tree.root.colour == Colour.BLACK
        child = tree.root.left if tree.root.left is not tree._nil else tree.root.right
        assert child.colour == Colour.RED


# ──────────────────────────────────────────────
#  Three Element Inserts (Rotation Triggers)
# ──────────────────────────────────────────────


class TestThreeInserts:
    def test_left_left_case(self):
        """30, 20, 10 -> right rotation; root becomes 20."""
        tree = RedBlackTree()
        tree.insert(30, "a")
        tree.insert(20, "b")
        tree.insert(10, "c")
        assert_rb_properties(tree)
        assert tree.root.key == 20

    def test_right_right_case(self):
        """10, 20, 30 -> left rotation; root becomes 20."""
        tree = RedBlackTree()
        tree.insert(10, "a")
        tree.insert(20, "b")
        tree.insert(30, "c")
        assert_rb_properties(tree)
        assert tree.root.key == 20

    def test_left_right_case(self):
        """30, 10, 20 -> left-right rotation; root becomes 20."""
        tree = RedBlackTree()
        tree.insert(30, "a")
        tree.insert(10, "b")
        tree.insert(20, "c")
        assert_rb_properties(tree)
        assert tree.root.key == 20

    def test_right_left_case(self):
        """10, 30, 20 -> right-left rotation; root becomes 20."""
        tree = RedBlackTree()
        tree.insert(10, "a")
        tree.insert(30, "b")
        tree.insert(20, "c")
        assert_rb_properties(tree)
        assert tree.root.key == 20

    def test_balanced_insert(self):
        """10, 5, 15 -> no rotation needed; root stays 10."""
        tree = RedBlackTree()
        tree.insert(10, "a")
        tree.insert(5, "b")
        tree.insert(15, "c")
        assert tree.root.key == 10
        assert tree.root.colour == Colour.BLACK
        assert_rb_properties(tree)


# ──────────────────────────────────────────────
#  Multiple Insertions
# ──────────────────────────────────────────────


class TestMultipleInserts:
    def test_ascending_order(self):
        tree = RedBlackTree()
        for k in range(1, 8):
            tree.insert(k, str(k))
        assert len(tree) == 7
        assert_rb_properties(tree)

    def test_descending_order(self):
        tree = RedBlackTree()
        for k in range(7, 0, -1):
            tree.insert(k, str(k))
        assert len(tree) == 7
        assert_rb_properties(tree)

    def test_mixed_order(self):
        tree = RedBlackTree()
        keys = [50, 25, 75, 12, 37, 62, 87, 6, 18, 31, 43]
        for k in keys:
            tree.insert(k, str(k))
        assert len(tree) == len(keys)
        for k in keys:
            assert tree.find(k) == str(k)
        assert_rb_properties(tree)

    def test_inorder_is_sorted(self):
        tree = RedBlackTree()
        keys = [30, 10, 50, 5, 20, 40, 60]
        for k in keys:
            tree.insert(k, str(k))
        pairs = collect_inorder(tree)
        assert [k for k, _ in pairs] == sorted(keys)

    def test_100_ascending(self):
        tree = RedBlackTree()
        for k in range(1, 101):
            tree.insert(k, k)
        assert len(tree) == 100
        assert_rb_properties(tree)

    def test_100_descending(self):
        tree = RedBlackTree()
        for k in range(100, 0, -1):
            tree.insert(k, k)
        assert len(tree) == 100
        assert_rb_properties(tree)

    def test_rb_properties_maintained_after_each_insert(self):
        tree = RedBlackTree()
        keys = [50, 25, 75, 12, 37, 62, 87, 6, 18, 31, 43, 56, 68, 81, 93]
        for k in keys:
            tree.insert(k, str(k))
            assert_rb_properties(tree)


# ──────────────────────────────────────────────
#  Insert Fixup Cases
# ──────────────────────────────────────────────


class TestInsertFixupCases:
    def test_uncle_red_recoloring(self):
        """Case 1: uncle is red -> recolor parent, uncle, grandparent."""
        tree = RedBlackTree()
        tree.insert(10, "a")
        tree.insert(5, "b")
        tree.insert(15, "c")
        tree.insert(3, "d")
        assert_rb_properties(tree)
        assert tree.root.colour == Colour.BLACK

    def test_recoloring_propagates_to_root(self):
        """Recoloring may propagate upward, requiring root re-blackening."""
        tree = RedBlackTree()
        for k in [10, 5, 15, 3, 7, 12, 20, 1]:
            tree.insert(k, str(k))
        assert_rb_properties(tree)

    def test_left_rotation_after_right_insertion(self):
        """Case 2+3: uncle black, node is right child -> rotate left then right."""
        tree = RedBlackTree()
        for k in [10, 5, 15, 7]:
            tree.insert(k, str(k))
        assert_rb_properties(tree)

    def test_right_rotation_after_left_insertion(self):
        """Case 2+3 (mirror): uncle black, node is left child -> rotate right then left."""
        tree = RedBlackTree()
        for k in [10, 5, 15, 12]:
            tree.insert(k, str(k))
        assert_rb_properties(tree)

    def test_deep_left_spine_forces_rotations(self):
        """Ascending inserts force repeated left rotations."""
        tree = RedBlackTree()
        for k in range(1, 16):
            tree.insert(k, k)
        assert_rb_properties(tree)

    def test_deep_right_spine_forces_rotations(self):
        """Descending inserts force repeated right rotations."""
        tree = RedBlackTree()
        for k in range(15, 0, -1):
            tree.insert(k, k)
        assert_rb_properties(tree)


# ──────────────────────────────────────────────
#  Find
# ──────────────────────────────────────────────


class TestFind:
    def test_find_root(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.find(10) == "ten"

    def test_find_left_child(self):
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        assert tree.find(5) == "5"

    def test_find_right_child(self):
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        assert tree.find(15) == "15"

    def test_find_deep_node(self):
        tree = RedBlackTree()
        for k in [50, 25, 75, 10, 30, 60, 90]:
            tree.insert(k, str(k))
        assert tree.find(10) == "10"
        assert tree.find(90) == "90"

    def test_find_nonexistent_key(self):
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        assert tree.find(7) is None

    def test_find_in_empty_tree(self):
        tree = RedBlackTree()
        assert tree.find(1) is None

    def test_find_smallest_key(self):
        tree = RedBlackTree()
        for k in [50, 25, 75, 10, 30]:
            tree.insert(k, str(k))
        assert tree.find(10) == "10"

    def test_find_largest_key(self):
        tree = RedBlackTree()
        for k in [50, 25, 75, 10, 30]:
            tree.insert(k, str(k))
        assert tree.find(75) == "75"

    def test_find_after_value_update(self):
        tree = RedBlackTree()
        tree.insert(10, "old")
        tree.insert(10, "new")
        assert tree.find(10) == "new"

    def test_find_all_inserted_keys(self):
        tree = RedBlackTree()
        keys = list(range(1, 21))
        for k in keys:
            tree.insert(k, k * 10)
        for k in keys:
            assert tree.find(k) == k * 10


# ──────────────────────────────────────────────
#  __iter__ (In-Order Traversal)
# ──────────────────────────────────────────────


class TestIter:
    def test_empty_tree_yields_nothing(self):
        tree = RedBlackTree()
        assert list(tree) == []

    def test_single_element(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert list(tree) == [(10, "ten")]

    def test_yields_key_value_tuples(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.insert(5, "five")
        result = list(tree)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)

    def test_sorted_order(self):
        tree = RedBlackTree()
        for k in [30, 10, 50, 5, 20, 40, 60]:
            tree.insert(k, str(k))
        keys = [k for k, _ in tree]
        assert keys == [5, 10, 20, 30, 40, 50, 60]

    def test_values_match_keys(self):
        tree = RedBlackTree()
        for k in [30, 10, 50, 5, 20]:
            tree.insert(k, k * 10)
        assert list(tree) == [(5, 50), (10, 100), (20, 200), (30, 300), (50, 500)]

    def test_matches_collect_inorder_helper(self):
        tree = RedBlackTree()
        for k in [50, 25, 75, 12, 37, 62, 87]:
            tree.insert(k, str(k))
        assert list(tree) == collect_inorder(tree)

    def test_count_matches_len(self):
        tree = RedBlackTree()
        for k in range(1, 16):
            tree.insert(k, k)
        assert len(list(tree)) == len(tree)

    def test_after_deletes(self):
        tree = RedBlackTree()
        for k in [10, 5, 15, 3, 7]:
            tree.insert(k, str(k))
        tree.delete(5)
        tree.delete(15)
        assert list(tree) == [(3, "3"), (7, "7"), (10, "10")]

    def test_after_delete_all(self):
        tree = RedBlackTree()
        tree.insert(1, "a")
        tree.delete(1)
        assert list(tree) == []

    def test_repeated_iteration_is_stable(self):
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        first = list(tree)
        second = list(tree)
        assert first == second

    def test_for_loop_unpacking(self):
        tree = RedBlackTree()
        tree.insert(1, "a")
        tree.insert(2, "b")
        tree.insert(3, "c")
        collected = {}
        for key, value in tree:
            collected[key] = value
        assert collected == {1: "a", 2: "b", 3: "c"}

    def test_dict_construction(self):
        tree = RedBlackTree()
        for k in [30, 10, 20]:
            tree.insert(k, str(k))
        assert dict(tree) == {10: "10", 20: "20", 30: "30"}

    def test_ascending_insertion_order(self):
        tree = RedBlackTree()
        for k in range(1, 8):
            tree.insert(k, str(k))
        keys = [k for k, _ in tree]
        assert keys == list(range(1, 8))

    def test_descending_insertion_order(self):
        tree = RedBlackTree()
        for k in range(7, 0, -1):
            tree.insert(k, str(k))
        keys = [k for k, _ in tree]
        assert keys == list(range(1, 8))

    def test_with_string_keys(self):
        tree = RedBlackTree()
        tree.insert("banana", 2)
        tree.insert("apple", 1)
        tree.insert("cherry", 3)
        assert list(tree) == [("apple", 1), ("banana", 2), ("cherry", 3)]

    def test_with_duplicate_inserts_reflects_latest_value(self):
        tree = RedBlackTree()
        tree.insert(1, "old")
        tree.insert(2, "b")
        tree.insert(1, "new")
        assert list(tree) == [(1, "new"), (2, "b")]

    def test_large_tree(self):
        tree = RedBlackTree()
        for k in range(500):
            tree.insert(k, k)
        result = list(tree)
        assert len(result) == 500
        assert result == [(k, k) for k in range(500)]


# ──────────────────────────────────────────────
#  Delete — Basic Operations
# ──────────────────────────────────────────────


class TestDeleteBasic:
    def test_delete_existing_returns_true(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.delete(10) is True

    def test_delete_nonexistent_returns_false(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.delete(99) is False

    def test_delete_from_empty_tree(self):
        tree = RedBlackTree()
        assert tree.delete(1) is False

    def test_delete_decrements_size(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.insert(20, "twenty")
        tree.delete(10)
        assert len(tree) == 1

    def test_find_returns_none_after_delete(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.delete(10)
        assert tree.find(10) is None

    def test_delete_only_element(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.delete(10) is True
        assert len(tree) == 0
        assert tree.find(10) is None

    def test_rb_properties_after_simple_delete(self):
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert_rb_properties(tree)

    def test_delete_same_key_twice(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.delete(10) is True
        assert tree.delete(10) is False

    def test_failed_delete_does_not_change_size(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.delete(999)
        assert len(tree) == 1


# ──────────────────────────────────────────────
#  Delete — Structural Cases
# ──────────────────────────────────────────────


class TestDeleteStructural:
    def test_delete_red_leaf(self):
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert tree.find(5) is None
        assert tree.find(10) == "10"
        assert tree.find(15) == "15"
        assert_rb_properties(tree)

    def test_delete_node_with_left_child_only(self):
        tree = RedBlackTree()
        for k in [10, 5, 15, 3]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert tree.find(5) is None
        assert tree.find(3) == "3"
        assert_rb_properties(tree)

    def test_delete_node_with_right_child_only(self):
        tree = RedBlackTree()
        for k in [10, 5, 15, 7]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert tree.find(5) is None
        assert tree.find(7) == "7"
        assert_rb_properties(tree)

    def test_delete_node_with_two_children(self):
        tree = RedBlackTree()
        for k in [10, 5, 15, 3, 7]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert tree.find(5) is None
        assert tree.find(3) == "3"
        assert tree.find(7) == "7"
        assert_rb_properties(tree)

    def test_delete_root_with_two_children(self):
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        tree.delete(10)
        assert tree.find(10) is None
        assert len(tree) == 2
        assert_rb_properties(tree)

    def test_delete_root_with_left_child_only(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.insert(5, "five")
        tree.delete(10)
        assert tree.find(10) is None
        assert tree.find(5) == "five"
        assert len(tree) == 1
        assert_rb_properties(tree)

    def test_delete_root_with_right_child_only(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.insert(15, "fifteen")
        tree.delete(10)
        assert tree.find(10) is None
        assert tree.find(15) == "fifteen"
        assert len(tree) == 1
        assert_rb_properties(tree)

    def test_delete_root_only_element(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.delete(10)
        assert len(tree) == 0
        assert tree.root is tree._nil or tree.root is None


# ──────────────────────────────────────────────
#  Delete — Fixup Cases (CLRS)
# ──────────────────────────────────────────────


class TestDeleteFixupCases:
    def test_case1_sibling_is_red(self):
        """Deleting a black node where sibling is red."""
        tree = RedBlackTree()
        for k in [10, 5, 15, 3, 7, 12, 20]:
            tree.insert(k, str(k))
        tree.delete(3)
        assert_rb_properties(tree)

    def test_case2_black_sibling_two_black_children(self):
        """Sibling is black with two black children -> recolor."""
        tree = RedBlackTree()
        for k in [10, 5, 15]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert_rb_properties(tree)

    def test_case3_black_sibling_far_child_red(self):
        """Sibling is black with far child red -> rotate and recolor."""
        tree = RedBlackTree()
        for k in [10, 5, 15, 20]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert_rb_properties(tree)

    def test_case4_black_sibling_near_child_red(self):
        """Sibling is black with near child red -> double rotation."""
        tree = RedBlackTree()
        for k in [10, 5, 15, 12]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert_rb_properties(tree)

    def test_delete_triggers_multiple_fixup_iterations(self):
        """Delete requiring fixup to propagate up the tree."""
        tree = RedBlackTree()
        for k in [20, 10, 30, 5, 15, 25, 35, 3, 7]:
            tree.insert(k, str(k))
        tree.delete(35)
        tree.delete(30)
        assert_rb_properties(tree)

    def test_successor_not_direct_child(self):
        """Delete node whose in-order successor is not its right child."""
        tree = RedBlackTree()
        for k in [20, 10, 30, 25, 35, 23]:
            tree.insert(k, str(k))
        tree.delete(20)
        assert tree.find(20) is None
        for k in [10, 30, 25, 35, 23]:
            assert tree.find(k) == str(k)
        assert_rb_properties(tree)

    def test_mirror_case1_sibling_is_red_left(self):
        """Mirror of case 1: node is right child, sibling (left) is red."""
        tree = RedBlackTree()
        for k in [10, 5, 15, 3, 7, 12, 20]:
            tree.insert(k, str(k))
        tree.delete(20)
        assert_rb_properties(tree)

    def test_mirror_case4_near_child_red_left(self):
        """Mirror of case 4: node is right child, sibling's near child is red."""
        tree = RedBlackTree()
        for k in [10, 5, 15, 7]:
            tree.insert(k, str(k))
        tree.delete(15)
        assert_rb_properties(tree)


# ──────────────────────────────────────────────
#  Delete All Elements
# ──────────────────────────────────────────────


class TestDeleteAll:
    def test_delete_all_ascending(self):
        tree = RedBlackTree()
        keys = list(range(1, 11))
        for k in keys:
            tree.insert(k, str(k))
        for k in keys:
            assert tree.delete(k) is True
            assert_rb_properties(tree)
        assert len(tree) == 0

    def test_delete_all_descending(self):
        tree = RedBlackTree()
        keys = list(range(1, 11))
        for k in keys:
            tree.insert(k, str(k))
        for k in reversed(keys):
            assert tree.delete(k) is True
            assert_rb_properties(tree)
        assert len(tree) == 0

    def test_delete_all_arbitrary_order(self):
        tree = RedBlackTree()
        keys = [50, 25, 75, 12, 37, 62, 87]
        for k in keys:
            tree.insert(k, str(k))
        delete_order = [37, 75, 12, 87, 50, 25, 62]
        for k in delete_order:
            assert tree.delete(k) is True
            assert_rb_properties(tree)
        assert len(tree) == 0

    def test_tree_reusable_after_delete_all(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        tree.delete(10)
        tree.insert(20, "twenty")
        assert tree.find(20) == "twenty"
        assert len(tree) == 1
        assert_rb_properties(tree)


# ──────────────────────────────────────────────
#  Interleaved Insert and Delete
# ──────────────────────────────────────────────


class TestInterleavedOperations:
    def test_insert_delete_reinsert_same_key(self):
        tree = RedBlackTree()
        tree.insert(10, "first")
        tree.delete(10)
        tree.insert(10, "second")
        assert tree.find(10) == "second"
        assert len(tree) == 1

    def test_interleaved_sequence(self):
        tree = RedBlackTree()
        tree.insert(10, "a")
        tree.insert(20, "b")
        tree.insert(30, "c")
        tree.delete(20)
        tree.insert(25, "d")
        tree.insert(15, "e")
        tree.delete(10)

        assert tree.find(10) is None
        assert tree.find(20) is None
        assert tree.find(30) == "c"
        assert tree.find(25) == "d"
        assert tree.find(15) == "e"
        assert len(tree) == 3
        assert_rb_properties(tree)

    def test_delete_evens_keep_odds(self):
        tree = RedBlackTree()
        for k in range(1, 21):
            tree.insert(k, k)
        for k in range(2, 21, 2):
            tree.delete(k)
        assert len(tree) == 10
        assert_rb_properties(tree)
        for k in range(1, 21, 2):
            assert tree.find(k) == k
        for k in range(2, 21, 2):
            assert tree.find(k) is None

    def test_alternating_insert_delete(self):
        tree = RedBlackTree()
        tree.insert(1, "a")
        tree.insert(2, "b")
        tree.delete(1)
        tree.insert(3, "c")
        tree.delete(2)
        tree.insert(4, "d")
        tree.delete(3)
        assert len(tree) == 1
        assert tree.find(4) == "d"
        assert_rb_properties(tree)

    def test_bulk_insert_then_selective_delete(self):
        tree = RedBlackTree()
        for k in range(1, 16):
            tree.insert(k, str(k))
        # Delete internal nodes
        for k in [8, 4, 12, 2, 6, 10, 14]:
            tree.delete(k)
            assert_rb_properties(tree)
        assert len(tree) == 8
        # Remaining keys: 1, 3, 5, 7, 9, 11, 13, 15
        for k in [1, 3, 5, 7, 9, 11, 13, 15]:
            assert tree.find(k) == str(k)


# ──────────────────────────────────────────────
#  Key Types
# ──────────────────────────────────────────────


class TestKeyTypes:
    def test_string_keys(self):
        tree = RedBlackTree()
        tree.insert("banana", 1)
        tree.insert("apple", 2)
        tree.insert("cherry", 3)
        assert tree.find("apple") == 2
        assert tree.find("banana") == 1
        assert tree.find("cherry") == 3
        assert_rb_properties(tree)

    def test_float_keys(self):
        tree = RedBlackTree()
        tree.insert(3.14, "pi")
        tree.insert(2.71, "e")
        tree.insert(1.41, "sqrt2")
        assert tree.find(3.14) == "pi"
        assert tree.find(2.71) == "e"
        assert_rb_properties(tree)

    def test_negative_integer_keys(self):
        tree = RedBlackTree()
        for k in [-5, -3, -1, 0, 2, 4]:
            tree.insert(k, str(k))
        assert len(tree) == 6
        for k in [-5, -3, -1, 0, 2, 4]:
            assert tree.find(k) == str(k)
        assert_rb_properties(tree)

    def test_tuple_keys(self):
        tree = RedBlackTree()
        tree.insert((1, 2), "a")
        tree.insert((0, 5), "b")
        tree.insert((2, 0), "c")
        assert tree.find((1, 2)) == "a"
        assert tree.find((0, 5)) == "b"
        assert_rb_properties(tree)

    def test_single_character_string_keys(self):
        tree = RedBlackTree()
        for ch in "zyxwvutsrqponmlkjihgfedcba":
            tree.insert(ch, ord(ch))
        assert len(tree) == 26
        assert tree.find("a") == ord("a")
        assert tree.find("z") == ord("z")
        assert_rb_properties(tree)


# ──────────────────────────────────────────────
#  Value Types
# ──────────────────────────────────────────────


class TestValueTypes:
    def test_none_value_is_ambiguous_with_not_found(self):
        """Storing None as a value is ambiguous since find() returns None for missing keys."""
        tree = RedBlackTree()
        tree.insert(1, None)
        # find returns None for both "found with None value" and "not found"
        assert tree.find(1) is None

    def test_dict_value(self):
        tree = RedBlackTree()
        tree.insert(1, {"name": "Alice"})
        assert tree.find(1) == {"name": "Alice"}

    def test_list_value(self):
        tree = RedBlackTree()
        tree.insert(1, [1, 2, 3])
        assert tree.find(1) == [1, 2, 3]

    def test_zero_value(self):
        tree = RedBlackTree()
        tree.insert(1, 0)
        assert tree.find(1) == 0

    def test_empty_string_value(self):
        tree = RedBlackTree()
        tree.insert(1, "")
        assert tree.find(1) == ""

    def test_boolean_value(self):
        tree = RedBlackTree()
        tree.insert(1, False)
        assert tree.find(1) is False


# ──────────────────────────────────────────────
#  __len__ Consistency
# ──────────────────────────────────────────────


class TestLen:
    def test_len_increments_on_each_insert(self):
        tree = RedBlackTree()
        for i in range(1, 6):
            tree.insert(i, i)
            assert len(tree) == i

    def test_len_decrements_on_each_delete(self):
        tree = RedBlackTree()
        for i in range(1, 6):
            tree.insert(i, i)
        for i in range(1, 6):
            tree.delete(i)
            assert len(tree) == 5 - i

    def test_len_unchanged_on_failed_delete(self):
        tree = RedBlackTree()
        tree.insert(1, "a")
        tree.delete(999)
        assert len(tree) == 1

    def test_len_unchanged_on_duplicate_insert(self):
        tree = RedBlackTree()
        tree.insert(1, "a")
        tree.insert(1, "b")
        assert len(tree) == 1

    def test_len_zero_after_all_deleted(self):
        tree = RedBlackTree()
        for k in range(10):
            tree.insert(k, k)
        for k in range(10):
            tree.delete(k)
        assert len(tree) == 0


# ──────────────────────────────────────────────
#  BST Ordering Invariant
# ──────────────────────────────────────────────


class TestBSTOrdering:
    def test_inorder_sorted_after_inserts(self):
        tree = RedBlackTree()
        keys = [42, 17, 88, 5, 29, 63, 95]
        for k in keys:
            tree.insert(k, str(k))
        pairs = collect_inorder(tree)
        assert [k for k, _ in pairs] == sorted(keys)

    def test_inorder_sorted_after_deletes(self):
        tree = RedBlackTree()
        keys = [42, 17, 88, 5, 29, 63, 95]
        for k in keys:
            tree.insert(k, str(k))
        tree.delete(42)
        tree.delete(5)
        remaining = sorted(set(keys) - {42, 5})
        pairs = collect_inorder(tree)
        assert [k for k, _ in pairs] == remaining

    def test_inorder_count_matches_size(self):
        tree = RedBlackTree()
        for k in [30, 10, 50, 5, 20, 40, 60]:
            tree.insert(k, str(k))
        pairs = collect_inorder(tree)
        assert len(pairs) == len(tree)


# ──────────────────────────────────────────────
#  Black-Height Property
# ──────────────────────────────────────────────


class TestBlackHeight:
    def test_consistent_after_sequential_inserts(self):
        tree = RedBlackTree()
        for k in range(1, 16):
            tree.insert(k, k)
            compute_black_height(tree, tree.root)

    def test_consistent_after_deletes(self):
        tree = RedBlackTree()
        for k in range(1, 16):
            tree.insert(k, k)
        for k in [8, 4, 12, 2, 6]:
            tree.delete(k)
            compute_black_height(tree, tree.root)


# ──────────────────────────────────────────────
#  Root Colour Property
# ──────────────────────────────────────────────


class TestRootColour:
    def test_root_black_after_single_insert(self):
        tree = RedBlackTree()
        tree.insert(1, "a")
        assert tree.root.colour == Colour.BLACK

    def test_root_black_after_every_insert(self):
        tree = RedBlackTree()
        for k in range(1, 20):
            tree.insert(k, k)
            assert tree.root.colour == Colour.BLACK, (
                f"Root not black after inserting {k}"
            )

    def test_root_black_after_deletes(self):
        tree = RedBlackTree()
        for k in [10, 5, 15, 3, 7]:
            tree.insert(k, str(k))
        for k in [10, 5, 15]:
            tree.delete(k)
            if len(tree) > 0:
                assert tree.root.colour == Colour.BLACK


# ──────────────────────────────────────────────
#  No Red-Red Violation Property
# ──────────────────────────────────────────────


class TestNoRedRedViolation:
    def test_no_red_red_after_sequential_inserts(self):
        tree = RedBlackTree()
        for k in range(1, 16):
            tree.insert(k, k)
            assert_no_red_red(tree)

    def test_no_red_red_after_deletes(self):
        tree = RedBlackTree()
        for k in range(1, 16):
            tree.insert(k, k)
        for k in [8, 4, 12, 2, 6, 10, 14]:
            tree.delete(k)
            assert_no_red_red(tree)


# ──────────────────────────────────────────────
#  Parent Pointer Consistency
# ──────────────────────────────────────────────


class TestParentPointers:
    def test_consistent_after_inserts(self):
        tree = RedBlackTree()
        for k in [10, 5, 15, 3, 7, 12, 20]:
            tree.insert(k, str(k))
        assert_parent_pointers(tree)

    def test_consistent_after_deletes(self):
        tree = RedBlackTree()
        for k in [10, 5, 15, 3, 7, 12, 20]:
            tree.insert(k, str(k))
        tree.delete(5)
        tree.delete(15)
        assert_parent_pointers(tree)

    def test_root_parent_is_nil(self):
        tree = RedBlackTree()
        tree.insert(10, "ten")
        assert tree.root.parent is tree._nil

    def test_consistent_after_rotations(self):
        """Ascending inserts force rotations; verify parent links survive."""
        tree = RedBlackTree()
        for k in range(1, 11):
            tree.insert(k, k)
        assert_parent_pointers(tree)


# ──────────────────────────────────────────────
#  Stress / Scale
# ──────────────────────────────────────────────


class TestStress:
    def test_1000_ascending_inserts(self):
        tree = RedBlackTree()
        for k in range(1000):
            tree.insert(k, k)
        assert len(tree) == 1000
        assert_rb_properties(tree)

    def test_insert_500_delete_evens(self):
        tree = RedBlackTree()
        for k in range(500):
            tree.insert(k, k)
        for k in range(0, 500, 2):
            tree.delete(k)
        assert len(tree) == 250
        assert_rb_properties(tree)
        for k in range(1, 500, 2):
            assert tree.find(k) == k

    def test_insert_then_delete_all_1000(self):
        tree = RedBlackTree()
        for k in range(1000):
            tree.insert(k, k)
        for k in range(1000):
            assert tree.delete(k) is True
        assert len(tree) == 0

    def test_find_all_in_large_tree(self):
        tree = RedBlackTree()
        for k in range(500):
            tree.insert(k, k * 2)
        for k in range(500):
            assert tree.find(k) == k * 2
        # Non-existent keys
        for k in range(500, 600):
            assert tree.find(k) is None


# ──────────────────────────────────────────────
#  Edge Cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    def test_insert_zero_key(self):
        tree = RedBlackTree()
        tree.insert(0, "zero")
        assert tree.find(0) == "zero"
        assert len(tree) == 1

    def test_insert_negative_keys(self):
        tree = RedBlackTree()
        tree.insert(-1, "neg1")
        tree.insert(-100, "neg100")
        assert tree.find(-1) == "neg1"
        assert tree.find(-100) == "neg100"
        assert_rb_properties(tree)

    def test_single_element_delete_leaves_empty_tree(self):
        tree = RedBlackTree()
        tree.insert(42, "x")
        tree.delete(42)
        assert len(tree) == 0
        assert tree.find(42) is None
        # Should be able to use tree after
        tree.insert(99, "y")
        assert tree.find(99) == "y"

    def test_delete_minimum_key(self):
        tree = RedBlackTree()
        for k in [20, 10, 30, 5, 15]:
            tree.insert(k, str(k))
        tree.delete(5)
        assert tree.find(5) is None
        assert_rb_properties(tree)

    def test_delete_maximum_key(self):
        tree = RedBlackTree()
        for k in [20, 10, 30, 5, 15]:
            tree.insert(k, str(k))
        tree.delete(30)
        assert tree.find(30) is None
        assert_rb_properties(tree)

    def test_many_duplicates_no_size_growth(self):
        tree = RedBlackTree()
        tree.insert(42, "v1")
        for i in range(100):
            tree.insert(42, f"v{i+2}")
        assert len(tree) == 1
        assert tree.find(42) == "v101"

    def test_delete_all_then_reinsert(self):
        tree = RedBlackTree()
        for k in range(1, 6):
            tree.insert(k, str(k))
        for k in range(1, 6):
            tree.delete(k)
        assert len(tree) == 0
        # Reinsert
        for k in range(10, 15):
            tree.insert(k, str(k))
        assert len(tree) == 5
        for k in range(10, 15):
            assert tree.find(k) == str(k)
        assert_rb_properties(tree)

    def test_tree_height_is_logarithmic(self):
        """For n=1000, height should be <= 2*log2(n+1) ~= 20."""
        tree = RedBlackTree()
        for k in range(1000):
            tree.insert(k, k)

        def height(node):
            if node is tree._nil or node is None:
                return 0
            return 1 + max(height(node.left), height(node.right))

        h = height(tree.root)
        import math
        max_allowed = 2 * math.log2(1001) + 1  # ~21
        assert h <= max_allowed, f"Height {h} exceeds expected max {max_allowed}"

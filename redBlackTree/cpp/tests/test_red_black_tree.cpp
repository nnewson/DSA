#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <red_black_tree/red_black_tree.hpp>

using red_black_tree::Colour;
using red_black_tree::RedBlackTree;

// Convenience alias — but avoid using inside GTest macros (comma in template args).
using IntTree = RedBlackTree<int, std::string>;
using IntIntTree = RedBlackTree<int, int>;
using StringIntTree = RedBlackTree<std::string, int>;

// ──────────────────────────────────────────────
//  Node accessor helpers (hide template commas)
// ──────────────────────────────────────────────

template <typename TreeT>
auto nk(const TreeT& /*tree*/, decltype(std::declval<TreeT>().root_node()) n)
{
    return TreeT::node_key(n);
}

template <typename TreeT>
auto nc(const TreeT& /*tree*/, decltype(std::declval<TreeT>().root_node()) n)
{
    return TreeT::node_colour(n);
}

template <typename TreeT>
auto nl(const TreeT& /*tree*/, decltype(std::declval<TreeT>().root_node()) n)
{
    return TreeT::node_left(n);
}

template <typename TreeT>
auto nr(const TreeT& /*tree*/, decltype(std::declval<TreeT>().root_node()) n)
{
    return TreeT::node_right(n);
}

template <typename TreeT>
auto np(const TreeT& /*tree*/, decltype(std::declval<TreeT>().root_node()) n)
{
    return TreeT::node_parent(n);
}

// ──────────────────────────────────────────────
//  Helpers for Red-Black Tree invariant checking
// ──────────────────────────────────────────────

template <typename K, typename V>
std::vector<std::pair<K, V>> collect_inorder(const RedBlackTree<K, V>& tree)
{
    return tree.collect_inorder();
}

template <typename K, typename V>
int compute_black_height(const RedBlackTree<K, V>& tree)
{
    auto nil = tree.nil_node();
    using TreeT = RedBlackTree<K, V>;
    using NodePtr = decltype(tree.root_node());

    std::function<int(NodePtr)> impl = [&](NodePtr node) -> int
    {
        if (node == nil)
            return 1;
        int left_bh = impl(TreeT::node_left(node));
        int right_bh = impl(TreeT::node_right(node));
        EXPECT_EQ(left_bh, right_bh);
        if (left_bh != right_bh)
            return -1;
        return left_bh + (TreeT::node_colour(node) == Colour::Black ? 1 : 0);
    };

    return impl(tree.root_node());
}

template <typename K, typename V>
void assert_no_red_red(const RedBlackTree<K, V>& tree)
{
    auto nil = tree.nil_node();
    using TreeT = RedBlackTree<K, V>;
    using NodePtr = decltype(tree.root_node());

    std::function<void(NodePtr)> impl = [&](NodePtr node)
    {
        if (node == nil)
            return;
        if (TreeT::node_colour(node) == Colour::Red)
        {
            auto left = TreeT::node_left(node);
            auto right = TreeT::node_right(node);
            if (left != nil)
            {
                auto col = TreeT::node_colour(left);
                EXPECT_EQ(col, Colour::Black);
            }
            if (right != nil)
            {
                auto col = TreeT::node_colour(right);
                EXPECT_EQ(col, Colour::Black);
            }
        }
        impl(TreeT::node_left(node));
        impl(TreeT::node_right(node));
    };

    impl(tree.root_node());
}

template <typename K, typename V>
void assert_parent_pointers(const RedBlackTree<K, V>& tree)
{
    auto nil = tree.nil_node();
    using TreeT = RedBlackTree<K, V>;
    using NodePtr = decltype(tree.root_node());

    std::function<void(NodePtr, NodePtr)> impl = [&](NodePtr node, NodePtr expected_parent)
    {
        if (node == nil)
            return;
        auto parent = TreeT::node_parent(node);
        EXPECT_EQ(parent, expected_parent);
        impl(TreeT::node_left(node), node);
        impl(TreeT::node_right(node), node);
    };

    impl(tree.root_node(), nil);
}

template <typename K, typename V>
void assert_rb_properties(const RedBlackTree<K, V>& tree)
{
    if (tree.root_node() == tree.nil_node())
        return;

    // Property 2: Root is black
    auto root_colour = RedBlackTree<K, V>::node_colour(tree.root_node());
    EXPECT_EQ(root_colour, Colour::Black) << "Root must be black";

    // Property 4: No red-red violations
    assert_no_red_red(tree);

    // Property 5: Consistent black-height
    compute_black_height(tree);

    // BST ordering invariant
    auto pairs = collect_inorder(tree);
    for (std::size_t i = 1; i < pairs.size(); ++i)
    {
        EXPECT_LT(pairs[i - 1].first, pairs[i].first) << "BST ordering violated";
    }

    // Parent pointer consistency
    assert_parent_pointers(tree);
}

// ──────────────────────────────────────────────
//  Colour Enum
// ──────────────────────────────────────────────

TEST(ColourEnum, RedIsNotBlack)
{
    EXPECT_NE(Colour::Red, Colour::Black);
}

// ──────────────────────────────────────────────
//  Empty Tree
// ──────────────────────────────────────────────

class EmptyTreeTest : public ::testing::Test
{
protected:
    IntTree tree;
};

TEST_F(EmptyTreeTest, InitialSizeIsZero)
{
    EXPECT_EQ(tree.size(), 0u);
}

TEST_F(EmptyTreeTest, IsEmpty)
{
    EXPECT_TRUE(tree.empty());
}

TEST_F(EmptyTreeTest, RootIsNilSentinel)
{
    EXPECT_EQ(tree.root_node(), tree.nil_node());
}

TEST_F(EmptyTreeTest, FindReturnsNullopt)
{
    EXPECT_FALSE(tree.find(42).has_value());
}

TEST_F(EmptyTreeTest, RemoveReturnsFalse)
{
    EXPECT_FALSE(tree.remove(42));
}

TEST_F(EmptyTreeTest, NilSentinelIsBlack)
{
    auto col = nc(tree, tree.nil_node());
    EXPECT_EQ(col, Colour::Black);
}

TEST_F(EmptyTreeTest, NilSentinelIsSelfReferential)
{
    auto nil = tree.nil_node();
    EXPECT_EQ(nl(tree, nil), nil);
    EXPECT_EQ(nr(tree, nil), nil);
    EXPECT_EQ(np(tree, nil), nil);
}

TEST_F(EmptyTreeTest, FindMultipleMissingKeys)
{
    for (int k : {0, -1, 100, 999})
    {
        EXPECT_FALSE(tree.find(k).has_value());
    }
}

TEST_F(EmptyTreeTest, RemoveMultipleMissingKeys)
{
    for (int k : {0, -1, 100})
    {
        EXPECT_FALSE(tree.remove(k));
    }
}

TEST_F(EmptyTreeTest, ContainsReturnsFalse)
{
    EXPECT_FALSE(tree.contains(1));
}

TEST_F(EmptyTreeTest, AtThrowsOutOfRange)
{
    EXPECT_THROW((void)tree.at(1), std::out_of_range);
}

TEST_F(EmptyTreeTest, IteratorBeginEqualsEnd)
{
    EXPECT_EQ(tree.begin(), tree.end());
}

// ──────────────────────────────────────────────
//  Single Element Insert
// ──────────────────────────────────────────────

class SingleInsertTest : public ::testing::Test
{
protected:
    IntTree tree;
    void SetUp() override
    {
        tree.add(10, "ten");
    }
};

TEST_F(SingleInsertTest, SizeIsOne)
{
    EXPECT_EQ(tree.size(), 1u);
}

TEST_F(SingleInsertTest, FindReturnsValue)
{
    auto result = tree.find(10);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->get(), "ten");
}

TEST_F(SingleInsertTest, RootIsBlack)
{
    auto col = nc(tree, tree.root_node());
    EXPECT_EQ(col, Colour::Black);
}

TEST_F(SingleInsertTest, RootKeyMatches)
{
    auto key = nk(tree, tree.root_node());
    EXPECT_EQ(key, 10);
}

TEST_F(SingleInsertTest, RootChildrenAreNil)
{
    auto root = tree.root_node();
    EXPECT_EQ(nl(tree, root), tree.nil_node());
    EXPECT_EQ(nr(tree, root), tree.nil_node());
}

TEST_F(SingleInsertTest, RootParentIsNil)
{
    EXPECT_EQ(np(tree, tree.root_node()), tree.nil_node());
}

TEST_F(SingleInsertTest, FindMissingKeyReturnsNullopt)
{
    EXPECT_FALSE(tree.find(999).has_value());
}

TEST_F(SingleInsertTest, RbProperties)
{
    assert_rb_properties(tree);
}

// ──────────────────────────────────────────────
//  Duplicate Key (Value Update)
// ──────────────────────────────────────────────

TEST(DuplicateInsert, UpdatesValue)
{
    IntTree tree;
    tree.add(10, "old");
    tree.add(10, "new");
    EXPECT_EQ(tree.find(10)->get(), "new");
}

TEST(DuplicateInsert, SizeUnchanged)
{
    IntTree tree;
    tree.add(10, "old");
    tree.add(10, "new");
    EXPECT_EQ(tree.size(), 1u);
}

TEST(DuplicateInsert, MultipleUpdates)
{
    IntTree tree;
    tree.add(5, "a");
    tree.add(5, "b");
    tree.add(5, "c");
    EXPECT_EQ(tree.find(5)->get(), "c");
    EXPECT_EQ(tree.size(), 1u);
}

TEST(DuplicateInsert, UpdatePreservesRbProperties)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    tree.add(5, "updated");
    assert_rb_properties(tree);
    EXPECT_EQ(tree.find(5)->get(), "updated");
}

// ──────────────────────────────────────────────
//  Two Element Inserts
// ──────────────────────────────────────────────

TEST(TwoInserts, SmallerThenLarger)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.add(5, "five");
    EXPECT_EQ(tree.size(), 2u);
    EXPECT_EQ(tree.find(5)->get(), "five");
    EXPECT_EQ(tree.find(10)->get(), "ten");
    assert_rb_properties(tree);
}

TEST(TwoInserts, LargerThenSmaller)
{
    IntTree tree;
    tree.add(5, "five");
    tree.add(10, "ten");
    EXPECT_EQ(tree.size(), 2u);
    assert_rb_properties(tree);
}

TEST(TwoInserts, ChildIsRed)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.add(5, "five");
    auto root = tree.root_node();
    auto nil = tree.nil_node();
    auto root_col = nc(tree, root);
    EXPECT_EQ(root_col, Colour::Black);
    auto left = nl(tree, root);
    auto right = nr(tree, root);
    auto child = (left != nil) ? left : right;
    auto child_col = nc(tree, child);
    EXPECT_EQ(child_col, Colour::Red);
}

// ──────────────────────────────────────────────
//  Three Element Inserts (Rotation Triggers)
// ──────────────────────────────────────────────

TEST(ThreeInserts, LeftLeftCase)
{
    IntTree tree;
    tree.add(30, "a");
    tree.add(20, "b");
    tree.add(10, "c");
    assert_rb_properties(tree);
    EXPECT_EQ(nk(tree, tree.root_node()), 20);
}

TEST(ThreeInserts, RightRightCase)
{
    IntTree tree;
    tree.add(10, "a");
    tree.add(20, "b");
    tree.add(30, "c");
    assert_rb_properties(tree);
    EXPECT_EQ(nk(tree, tree.root_node()), 20);
}

TEST(ThreeInserts, LeftRightCase)
{
    IntTree tree;
    tree.add(30, "a");
    tree.add(10, "b");
    tree.add(20, "c");
    assert_rb_properties(tree);
    EXPECT_EQ(nk(tree, tree.root_node()), 20);
}

TEST(ThreeInserts, RightLeftCase)
{
    IntTree tree;
    tree.add(10, "a");
    tree.add(30, "b");
    tree.add(20, "c");
    assert_rb_properties(tree);
    EXPECT_EQ(nk(tree, tree.root_node()), 20);
}

TEST(ThreeInserts, BalancedInsert)
{
    IntTree tree;
    tree.add(10, "a");
    tree.add(5, "b");
    tree.add(15, "c");
    EXPECT_EQ(nk(tree, tree.root_node()), 10);
    auto col = nc(tree, tree.root_node());
    EXPECT_EQ(col, Colour::Black);
    assert_rb_properties(tree);
}

// ──────────────────────────────────────────────
//  Multiple Insertions
// ──────────────────────────────────────────────

TEST(MultipleInserts, AscendingOrder)
{
    IntTree tree;
    for (int k = 1; k <= 7; ++k)
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.size(), 7u);
    assert_rb_properties(tree);
}

TEST(MultipleInserts, DescendingOrder)
{
    IntTree tree;
    for (int k = 7; k >= 1; --k)
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.size(), 7u);
    assert_rb_properties(tree);
}

TEST(MultipleInserts, MixedOrder)
{
    IntTree tree;
    std::vector<int> keys = {50, 25, 75, 12, 37, 62, 87, 6, 18, 31, 43};
    for (int k : keys)
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.size(), keys.size());
    for (int k : keys)
        EXPECT_EQ(tree.find(k)->get(), std::to_string(k));
    assert_rb_properties(tree);
}

TEST(MultipleInserts, InorderIsSorted)
{
    IntTree tree;
    std::vector<int> keys = {30, 10, 50, 5, 20, 40, 60};
    for (int k : keys)
        tree.add(k, std::to_string(k));
    auto pairs = collect_inorder(tree);
    std::sort(keys.begin(), keys.end());
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        EXPECT_EQ(pairs[i].first, keys[i]);
    }
}

TEST(MultipleInserts, HundredAscending)
{
    IntIntTree tree;
    for (int k = 1; k <= 100; ++k)
        tree.add(k, k);
    EXPECT_EQ(tree.size(), 100u);
    assert_rb_properties(tree);
}

TEST(MultipleInserts, HundredDescending)
{
    IntIntTree tree;
    for (int k = 100; k >= 1; --k)
        tree.add(k, k);
    EXPECT_EQ(tree.size(), 100u);
    assert_rb_properties(tree);
}

TEST(MultipleInserts, RbPropertiesMaintainedAfterEachInsert)
{
    IntTree tree;
    std::vector<int> keys = {50, 25, 75, 12, 37, 62, 87, 6, 18, 31, 43, 56, 68, 81, 93};
    for (int k : keys)
    {
        tree.add(k, std::to_string(k));
        assert_rb_properties(tree);
    }
}

// ──────────────────────────────────────────────
//  Insert Fixup Cases
// ──────────────────────────────────────────────

TEST(InsertFixupCases, UncleRedRecoloring)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3})
        tree.add(k, std::to_string(k));
    assert_rb_properties(tree);
    auto col = nc(tree, tree.root_node());
    EXPECT_EQ(col, Colour::Black);
}

TEST(InsertFixupCases, RecoloringPropagatesToRoot)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3, 7, 12, 20, 1})
        tree.add(k, std::to_string(k));
    assert_rb_properties(tree);
}

TEST(InsertFixupCases, LeftRotationAfterRightInsertion)
{
    IntTree tree;
    for (int k : {10, 5, 15, 7})
        tree.add(k, std::to_string(k));
    assert_rb_properties(tree);
}

TEST(InsertFixupCases, RightRotationAfterLeftInsertion)
{
    IntTree tree;
    for (int k : {10, 5, 15, 12})
        tree.add(k, std::to_string(k));
    assert_rb_properties(tree);
}

TEST(InsertFixupCases, DeepLeftSpineForcesRotations)
{
    IntIntTree tree;
    for (int k = 1; k <= 15; ++k)
        tree.add(k, k);
    assert_rb_properties(tree);
}

TEST(InsertFixupCases, DeepRightSpineForcesRotations)
{
    IntIntTree tree;
    for (int k = 15; k >= 1; --k)
        tree.add(k, k);
    assert_rb_properties(tree);
}

// ──────────────────────────────────────────────
//  Contains
// ──────────────────────────────────────────────

TEST(Contains, EmptyTree)
{
    IntTree tree;
    EXPECT_FALSE(tree.contains(1));
}

TEST(Contains, ExistingKey)
{
    IntTree tree;
    tree.add(5, "five");
    tree.add(3, "three");
    tree.add(7, "seven");
    EXPECT_TRUE(tree.contains(5));
    EXPECT_TRUE(tree.contains(3));
    EXPECT_TRUE(tree.contains(7));
}

TEST(Contains, MissingKey)
{
    IntTree tree;
    tree.add(5, "five");
    EXPECT_FALSE(tree.contains(4));
    EXPECT_FALSE(tree.contains(6));
}

TEST(Contains, AfterRemove)
{
    IntTree tree;
    tree.add(5, "five");
    tree.add(3, "three");
    tree.remove(5);
    EXPECT_FALSE(tree.contains(5));
    EXPECT_TRUE(tree.contains(3));
}

// ──────────────────────────────────────────────
//  At / operator[]
// ──────────────────────────────────────────────

TEST(At, ExistingKey)
{
    IntTree tree;
    tree.add(5, "five");
    tree.add(3, "three");
    EXPECT_EQ(tree.at(5), "five");
    EXPECT_EQ(tree.at(3), "three");
}

TEST(At, MissingKeyThrows)
{
    IntTree tree;
    tree.add(5, "five");
    EXPECT_THROW((void)tree.at(99), std::out_of_range);
}

TEST(At, EmptyTreeThrows)
{
    IntTree tree;
    EXPECT_THROW((void)tree.at(1), std::out_of_range);
}

TEST(At, UpdatedValue)
{
    IntTree tree;
    tree.add(5, "five");
    tree.add(5, "updated");
    EXPECT_EQ(tree.at(5), "updated");
}

TEST(SubscriptOperator, ExistingKey)
{
    IntTree tree;
    tree.add(5, "five");
    EXPECT_EQ(tree[5], "five");
}

TEST(SubscriptOperator, MissingKeyThrows)
{
    IntTree tree;
    EXPECT_THROW(tree[99], std::out_of_range);
}

// ──────────────────────────────────────────────
//  Find
// ──────────────────────────────────────────────

TEST(Find, Root)
{
    IntTree tree;
    tree.add(10, "ten");
    EXPECT_EQ(tree.find(10)->get(), "ten");
}

TEST(Find, LeftChild)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.find(5)->get(), "5");
}

TEST(Find, RightChild)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.find(15)->get(), "15");
}

TEST(Find, DeepNode)
{
    IntTree tree;
    for (int k : {50, 25, 75, 10, 30, 60, 90})
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.find(10)->get(), "10");
    EXPECT_EQ(tree.find(90)->get(), "90");
}

TEST(Find, NonexistentKey)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    EXPECT_FALSE(tree.find(7).has_value());
}

TEST(Find, EmptyTree)
{
    IntTree tree;
    EXPECT_FALSE(tree.find(1).has_value());
}

TEST(Find, SmallestKey)
{
    IntTree tree;
    for (int k : {50, 25, 75, 10, 30})
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.find(10)->get(), "10");
}

TEST(Find, LargestKey)
{
    IntTree tree;
    for (int k : {50, 25, 75, 10, 30})
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.find(75)->get(), "75");
}

TEST(Find, AfterValueUpdate)
{
    IntTree tree;
    tree.add(10, "old");
    tree.add(10, "new");
    EXPECT_EQ(tree.find(10)->get(), "new");
}

TEST(Find, AllInsertedKeys)
{
    IntIntTree tree;
    for (int k = 1; k <= 20; ++k)
        tree.add(k, k * 10);
    for (int k = 1; k <= 20; ++k)
        EXPECT_EQ(tree.find(k)->get(), k * 10);
}

// ──────────────────────────────────────────────
//  Iterator (In-Order Traversal)
// ──────────────────────────────────────────────

TEST(Iter, EmptyTreeYieldsNothing)
{
    IntTree tree;
    EXPECT_EQ(tree.begin(), tree.end());
    EXPECT_TRUE(tree.collect_inorder().empty());
}

TEST(Iter, SingleElement)
{
    IntTree tree;
    tree.add(10, "ten");
    auto pairs = tree.collect_inorder();
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].first, 10);
    EXPECT_EQ(pairs[0].second, "ten");
}

TEST(Iter, SortedOrder)
{
    IntTree tree;
    for (int k : {30, 10, 50, 5, 20, 40, 60})
        tree.add(k, std::to_string(k));
    auto pairs = tree.collect_inorder();
    std::vector<int> keys;
    for (const auto& [k, v] : pairs)
        keys.push_back(k);
    std::vector<int> expected = {5, 10, 20, 30, 40, 50, 60};
    EXPECT_EQ(keys, expected);
}

TEST(Iter, ValuesMatchKeys)
{
    IntIntTree tree;
    for (int k : {30, 10, 50, 5, 20})
        tree.add(k, k * 10);
    auto pairs = tree.collect_inorder();
    ASSERT_EQ(pairs.size(), 5u);
    EXPECT_EQ(pairs[0].first, 5);
    EXPECT_EQ(pairs[0].second, 50);
    EXPECT_EQ(pairs[1].first, 10);
    EXPECT_EQ(pairs[1].second, 100);
    EXPECT_EQ(pairs[2].first, 20);
    EXPECT_EQ(pairs[2].second, 200);
    EXPECT_EQ(pairs[3].first, 30);
    EXPECT_EQ(pairs[3].second, 300);
    EXPECT_EQ(pairs[4].first, 50);
    EXPECT_EQ(pairs[4].second, 500);
}

TEST(Iter, CountMatchesSize)
{
    IntIntTree tree;
    for (int k = 1; k <= 15; ++k)
        tree.add(k, k);
    EXPECT_EQ(tree.collect_inorder().size(), tree.size());
}

TEST(Iter, AfterRemoves)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3, 7})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    tree.remove(15);
    auto pairs = tree.collect_inorder();
    ASSERT_EQ(pairs.size(), 3u);
    EXPECT_EQ(pairs[0].first, 3);
    EXPECT_EQ(pairs[1].first, 7);
    EXPECT_EQ(pairs[2].first, 10);
}

TEST(Iter, AfterRemoveAll)
{
    IntTree tree;
    tree.add(1, "a");
    tree.remove(1);
    EXPECT_TRUE(tree.collect_inorder().empty());
}

TEST(Iter, RepeatedIterationIsStable)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    auto first = tree.collect_inorder();
    auto second = tree.collect_inorder();
    EXPECT_EQ(first, second);
}

TEST(Iter, RangeForLoop)
{
    IntTree tree;
    tree.add(1, "a");
    tree.add(2, "b");
    tree.add(3, "c");
    std::vector<std::pair<int, std::string>> collected;
    for (auto [key, value] : tree)
    {
        collected.emplace_back(key, value);
    }
    ASSERT_EQ(collected.size(), 3u);
    EXPECT_EQ(collected[0].first, 1);
    EXPECT_EQ(collected[1].first, 2);
    EXPECT_EQ(collected[2].first, 3);
}

TEST(Iter, AscendingInsertionOrder)
{
    IntTree tree;
    for (int k = 1; k <= 7; ++k)
        tree.add(k, std::to_string(k));
    auto pairs = tree.collect_inorder();
    for (int i = 0; i < 7; ++i)
        EXPECT_EQ(pairs[i].first, i + 1);
}

TEST(Iter, DescendingInsertionOrder)
{
    IntTree tree;
    for (int k = 7; k >= 1; --k)
        tree.add(k, std::to_string(k));
    auto pairs = tree.collect_inorder();
    for (int i = 0; i < 7; ++i)
        EXPECT_EQ(pairs[i].first, i + 1);
}

TEST(Iter, WithStringKeys)
{
    StringIntTree tree;
    tree.add("banana", 2);
    tree.add("apple", 1);
    tree.add("cherry", 3);
    auto pairs = tree.collect_inorder();
    ASSERT_EQ(pairs.size(), 3u);
    EXPECT_EQ(pairs[0].first, "apple");
    EXPECT_EQ(pairs[1].first, "banana");
    EXPECT_EQ(pairs[2].first, "cherry");
}

TEST(Iter, DuplicateInsertsReflectsLatestValue)
{
    IntTree tree;
    tree.add(1, "old");
    tree.add(2, "b");
    tree.add(1, "new");
    auto pairs = tree.collect_inorder();
    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].second, "new");
    EXPECT_EQ(pairs[1].second, "b");
}

TEST(Iter, LargeTree)
{
    IntIntTree tree;
    for (int k = 0; k < 500; ++k)
        tree.add(k, k);
    auto result = tree.collect_inorder();
    EXPECT_EQ(result.size(), 500u);
    for (int k = 0; k < 500; ++k)
    {
        EXPECT_EQ(result[k].first, k);
        EXPECT_EQ(result[k].second, k);
    }
}

// ──────────────────────────────────────────────
//  Delete — Basic Operations
// ──────────────────────────────────────────────

TEST(DeleteBasic, ExistingReturnsTrue)
{
    IntTree tree;
    tree.add(10, "ten");
    EXPECT_TRUE(tree.remove(10));
}

TEST(DeleteBasic, NonexistentReturnsFalse)
{
    IntTree tree;
    tree.add(10, "ten");
    EXPECT_FALSE(tree.remove(99));
}

TEST(DeleteBasic, FromEmptyTree)
{
    IntTree tree;
    EXPECT_FALSE(tree.remove(1));
}

TEST(DeleteBasic, DecrementsSize)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.add(20, "twenty");
    tree.remove(10);
    EXPECT_EQ(tree.size(), 1u);
}

TEST(DeleteBasic, FindReturnsNulloptAfterDelete)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.remove(10);
    EXPECT_FALSE(tree.find(10).has_value());
}

TEST(DeleteBasic, OnlyElement)
{
    IntTree tree;
    tree.add(10, "ten");
    EXPECT_TRUE(tree.remove(10));
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_FALSE(tree.find(10).has_value());
}

TEST(DeleteBasic, RbPropertiesAfterSimpleDelete)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    assert_rb_properties(tree);
}

TEST(DeleteBasic, SameKeyTwice)
{
    IntTree tree;
    tree.add(10, "ten");
    EXPECT_TRUE(tree.remove(10));
    EXPECT_FALSE(tree.remove(10));
}

TEST(DeleteBasic, FailedDeleteDoesNotChangeSize)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.remove(999);
    EXPECT_EQ(tree.size(), 1u);
}

// ──────────────────────────────────────────────
//  Delete — Structural Cases
// ──────────────────────────────────────────────

TEST(DeleteStructural, RedLeaf)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    EXPECT_FALSE(tree.find(5).has_value());
    EXPECT_EQ(tree.find(10)->get(), "10");
    EXPECT_EQ(tree.find(15)->get(), "15");
    assert_rb_properties(tree);
}

TEST(DeleteStructural, NodeWithLeftChildOnly)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    EXPECT_FALSE(tree.find(5).has_value());
    EXPECT_EQ(tree.find(3)->get(), "3");
    assert_rb_properties(tree);
}

TEST(DeleteStructural, NodeWithRightChildOnly)
{
    IntTree tree;
    for (int k : {10, 5, 15, 7})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    EXPECT_FALSE(tree.find(5).has_value());
    EXPECT_EQ(tree.find(7)->get(), "7");
    assert_rb_properties(tree);
}

TEST(DeleteStructural, NodeWithTwoChildren)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3, 7})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    EXPECT_FALSE(tree.find(5).has_value());
    EXPECT_EQ(tree.find(3)->get(), "3");
    EXPECT_EQ(tree.find(7)->get(), "7");
    assert_rb_properties(tree);
}

TEST(DeleteStructural, RootWithTwoChildren)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    tree.remove(10);
    EXPECT_FALSE(tree.find(10).has_value());
    EXPECT_EQ(tree.size(), 2u);
    assert_rb_properties(tree);
}

TEST(DeleteStructural, RootWithLeftChildOnly)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.add(5, "five");
    tree.remove(10);
    EXPECT_FALSE(tree.find(10).has_value());
    EXPECT_EQ(tree.find(5)->get(), "five");
    EXPECT_EQ(tree.size(), 1u);
    assert_rb_properties(tree);
}

TEST(DeleteStructural, RootWithRightChildOnly)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.add(15, "fifteen");
    tree.remove(10);
    EXPECT_FALSE(tree.find(10).has_value());
    EXPECT_EQ(tree.find(15)->get(), "fifteen");
    EXPECT_EQ(tree.size(), 1u);
    assert_rb_properties(tree);
}

TEST(DeleteStructural, RootOnlyElement)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.remove(10);
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_EQ(tree.root_node(), tree.nil_node());
}

// ──────────────────────────────────────────────
//  Delete — Fixup Cases (CLRS)
// ──────────────────────────────────────────────

TEST(DeleteFixupCases, Case1SiblingIsRed)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3, 7, 12, 20})
        tree.add(k, std::to_string(k));
    tree.remove(3);
    assert_rb_properties(tree);
}

TEST(DeleteFixupCases, Case2BlackSiblingTwoBlackChildren)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    assert_rb_properties(tree);
}

TEST(DeleteFixupCases, Case3BlackSiblingFarChildRed)
{
    IntTree tree;
    for (int k : {10, 5, 15, 20})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    assert_rb_properties(tree);
}

TEST(DeleteFixupCases, Case4BlackSiblingNearChildRed)
{
    IntTree tree;
    for (int k : {10, 5, 15, 12})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    assert_rb_properties(tree);
}

TEST(DeleteFixupCases, MultipleFixupIterations)
{
    IntTree tree;
    for (int k : {20, 10, 30, 5, 15, 25, 35, 3, 7})
        tree.add(k, std::to_string(k));
    tree.remove(35);
    tree.remove(30);
    assert_rb_properties(tree);
}

TEST(DeleteFixupCases, SuccessorNotDirectChild)
{
    IntTree tree;
    for (int k : {20, 10, 30, 25, 35, 23})
        tree.add(k, std::to_string(k));
    tree.remove(20);
    EXPECT_FALSE(tree.find(20).has_value());
    for (int k : {10, 30, 25, 35, 23})
        EXPECT_EQ(tree.find(k)->get(), std::to_string(k));
    assert_rb_properties(tree);
}

TEST(DeleteFixupCases, MirrorCase1SiblingIsRedLeft)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3, 7, 12, 20})
        tree.add(k, std::to_string(k));
    tree.remove(20);
    assert_rb_properties(tree);
}

TEST(DeleteFixupCases, MirrorCase4NearChildRedLeft)
{
    IntTree tree;
    for (int k : {10, 5, 15, 7})
        tree.add(k, std::to_string(k));
    tree.remove(15);
    assert_rb_properties(tree);
}

// ──────────────────────────────────────────────
//  Delete All Elements
// ──────────────────────────────────────────────

TEST(DeleteAll, Ascending)
{
    IntTree tree;
    for (int k = 1; k <= 10; ++k)
        tree.add(k, std::to_string(k));
    for (int k = 1; k <= 10; ++k)
    {
        EXPECT_TRUE(tree.remove(k));
        assert_rb_properties(tree);
    }
    EXPECT_EQ(tree.size(), 0u);
}

TEST(DeleteAll, Descending)
{
    IntTree tree;
    for (int k = 1; k <= 10; ++k)
        tree.add(k, std::to_string(k));
    for (int k = 10; k >= 1; --k)
    {
        EXPECT_TRUE(tree.remove(k));
        assert_rb_properties(tree);
    }
    EXPECT_EQ(tree.size(), 0u);
}

TEST(DeleteAll, ArbitraryOrder)
{
    IntTree tree;
    for (int k : {50, 25, 75, 12, 37, 62, 87})
        tree.add(k, std::to_string(k));
    for (int k : {37, 75, 12, 87, 50, 25, 62})
    {
        EXPECT_TRUE(tree.remove(k));
        assert_rb_properties(tree);
    }
    EXPECT_EQ(tree.size(), 0u);
}

TEST(DeleteAll, TreeReusableAfterDeleteAll)
{
    IntTree tree;
    tree.add(10, "ten");
    tree.remove(10);
    tree.add(20, "twenty");
    EXPECT_EQ(tree.find(20)->get(), "twenty");
    EXPECT_EQ(tree.size(), 1u);
    assert_rb_properties(tree);
}

// ──────────────────────────────────────────────
//  Interleaved Insert and Delete
// ──────────────────────────────────────────────

TEST(InterleavedOps, InsertDeleteReinsertSameKey)
{
    IntTree tree;
    tree.add(10, "first");
    tree.remove(10);
    tree.add(10, "second");
    EXPECT_EQ(tree.find(10)->get(), "second");
    EXPECT_EQ(tree.size(), 1u);
}

TEST(InterleavedOps, InterleavedSequence)
{
    IntTree tree;
    tree.add(10, "a");
    tree.add(20, "b");
    tree.add(30, "c");
    tree.remove(20);
    tree.add(25, "d");
    tree.add(15, "e");
    tree.remove(10);

    EXPECT_FALSE(tree.find(10).has_value());
    EXPECT_FALSE(tree.find(20).has_value());
    EXPECT_EQ(tree.find(30)->get(), "c");
    EXPECT_EQ(tree.find(25)->get(), "d");
    EXPECT_EQ(tree.find(15)->get(), "e");
    EXPECT_EQ(tree.size(), 3u);
    assert_rb_properties(tree);
}

TEST(InterleavedOps, DeleteEvensKeepOdds)
{
    IntIntTree tree;
    for (int k = 1; k <= 20; ++k)
        tree.add(k, k);
    for (int k = 2; k <= 20; k += 2)
        tree.remove(k);
    EXPECT_EQ(tree.size(), 10u);
    assert_rb_properties(tree);
    for (int k = 1; k <= 20; k += 2)
        EXPECT_EQ(tree.find(k)->get(), k);
    for (int k = 2; k <= 20; k += 2)
        EXPECT_FALSE(tree.find(k).has_value());
}

TEST(InterleavedOps, AlternatingInsertDelete)
{
    IntTree tree;
    tree.add(1, "a");
    tree.add(2, "b");
    tree.remove(1);
    tree.add(3, "c");
    tree.remove(2);
    tree.add(4, "d");
    tree.remove(3);
    EXPECT_EQ(tree.size(), 1u);
    EXPECT_EQ(tree.find(4)->get(), "d");
    assert_rb_properties(tree);
}

TEST(InterleavedOps, BulkInsertThenSelectiveDelete)
{
    IntTree tree;
    for (int k = 1; k <= 15; ++k)
        tree.add(k, std::to_string(k));
    for (int k : {8, 4, 12, 2, 6, 10, 14})
    {
        tree.remove(k);
        assert_rb_properties(tree);
    }
    EXPECT_EQ(tree.size(), 8u);
    for (int k : {1, 3, 5, 7, 9, 11, 13, 15})
    {
        EXPECT_EQ(tree.find(k)->get(), std::to_string(k));
    }
}

// ──────────────────────────────────────────────
//  Key Types
// ──────────────────────────────────────────────

TEST(KeyTypes, StringKeys)
{
    StringIntTree tree;
    tree.add("banana", 1);
    tree.add("apple", 2);
    tree.add("cherry", 3);
    EXPECT_EQ(tree.find("apple")->get(), 2);
    EXPECT_EQ(tree.find("banana")->get(), 1);
    EXPECT_EQ(tree.find("cherry")->get(), 3);
    assert_rb_properties(tree);
}

TEST(KeyTypes, DoubleKeys)
{
    RedBlackTree<double, std::string> tree;
    tree.add(3.14, "pi");
    tree.add(2.71, "e");
    tree.add(1.41, "sqrt2");
    EXPECT_EQ(tree.find(3.14)->get(), "pi");
    EXPECT_EQ(tree.find(2.71)->get(), "e");
    assert_rb_properties(tree);
}

TEST(KeyTypes, NegativeIntegerKeys)
{
    IntTree tree;
    for (int k : {-5, -3, -1, 0, 2, 4})
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.size(), 6u);
    for (int k : {-5, -3, -1, 0, 2, 4})
        EXPECT_EQ(tree.find(k)->get(), std::to_string(k));
    assert_rb_properties(tree);
}

TEST(KeyTypes, SingleCharStringKeys)
{
    StringIntTree tree;
    std::string alpha = "zyxwvutsrqponmlkjihgfedcba";
    for (char ch : alpha)
    {
        std::string s(1, ch);
        tree.add(s, static_cast<int>(ch));
    }
    EXPECT_EQ(tree.size(), 26u);
    EXPECT_EQ(tree.find("a")->get(), static_cast<int>('a'));
    EXPECT_EQ(tree.find("z")->get(), static_cast<int>('z'));
    assert_rb_properties(tree);
}

// ──────────────────────────────────────────────
//  Value Types
// ──────────────────────────────────────────────

TEST(ValueTypes, ZeroValue)
{
    IntIntTree tree;
    tree.add(1, 0);
    EXPECT_EQ(tree.find(1)->get(), 0);
}

TEST(ValueTypes, EmptyStringValue)
{
    IntTree tree;
    tree.add(1, "");
    EXPECT_EQ(tree.find(1)->get(), "");
}

TEST(ValueTypes, BooleanValue)
{
    RedBlackTree<int, bool> tree;
    tree.add(1, false);
    EXPECT_EQ(tree.find(1)->get(), false);
}

TEST(ValueTypes, VectorValue)
{
    RedBlackTree<int, std::vector<int>> tree;
    tree.add(1, {1, 2, 3});
    std::vector<int> expected = {1, 2, 3};
    EXPECT_EQ(tree.find(1)->get(), expected);
}

// ──────────────────────────────────────────────
//  Size Consistency
// ──────────────────────────────────────────────

TEST(Len, IncrementsOnEachInsert)
{
    IntIntTree tree;
    for (int i = 1; i <= 5; ++i)
    {
        tree.add(i, i);
        EXPECT_EQ(tree.size(), static_cast<std::size_t>(i));
    }
}

TEST(Len, DecrementsOnEachDelete)
{
    IntIntTree tree;
    for (int i = 1; i <= 5; ++i)
        tree.add(i, i);
    for (int i = 1; i <= 5; ++i)
    {
        tree.remove(i);
        EXPECT_EQ(tree.size(), static_cast<std::size_t>(5 - i));
    }
}

TEST(Len, UnchangedOnFailedDelete)
{
    IntTree tree;
    tree.add(1, "a");
    tree.remove(999);
    EXPECT_EQ(tree.size(), 1u);
}

TEST(Len, UnchangedOnDuplicateInsert)
{
    IntTree tree;
    tree.add(1, "a");
    tree.add(1, "b");
    EXPECT_EQ(tree.size(), 1u);
}

TEST(Len, ZeroAfterAllDeleted)
{
    IntIntTree tree;
    for (int k = 0; k < 10; ++k)
        tree.add(k, k);
    for (int k = 0; k < 10; ++k)
        tree.remove(k);
    EXPECT_EQ(tree.size(), 0u);
}

// ──────────────────────────────────────────────
//  BST Ordering Invariant
// ──────────────────────────────────────────────

TEST(BSTOrdering, InorderSortedAfterInserts)
{
    IntTree tree;
    std::vector<int> keys = {42, 17, 88, 5, 29, 63, 95};
    for (int k : keys)
        tree.add(k, std::to_string(k));
    auto pairs = collect_inorder(tree);
    std::sort(keys.begin(), keys.end());
    for (std::size_t i = 0; i < keys.size(); ++i)
        EXPECT_EQ(pairs[i].first, keys[i]);
}

TEST(BSTOrdering, InorderSortedAfterDeletes)
{
    IntTree tree;
    std::vector<int> keys = {42, 17, 88, 5, 29, 63, 95};
    for (int k : keys)
        tree.add(k, std::to_string(k));
    tree.remove(42);
    tree.remove(5);
    auto pairs = collect_inorder(tree);
    std::vector<int> remaining = {17, 29, 63, 88, 95};
    for (std::size_t i = 0; i < remaining.size(); ++i)
        EXPECT_EQ(pairs[i].first, remaining[i]);
}

TEST(BSTOrdering, InorderCountMatchesSize)
{
    IntTree tree;
    for (int k : {30, 10, 50, 5, 20, 40, 60})
        tree.add(k, std::to_string(k));
    EXPECT_EQ(collect_inorder(tree).size(), tree.size());
}

// ──────────────────────────────────────────────
//  Black-Height Property
// ──────────────────────────────────────────────

TEST(BlackHeight, ConsistentAfterSequentialInserts)
{
    IntIntTree tree;
    for (int k = 1; k <= 15; ++k)
    {
        tree.add(k, k);
        compute_black_height(tree);
    }
}

TEST(BlackHeight, ConsistentAfterDeletes)
{
    IntIntTree tree;
    for (int k = 1; k <= 15; ++k)
        tree.add(k, k);
    for (int k : {8, 4, 12, 2, 6})
    {
        tree.remove(k);
        compute_black_height(tree);
    }
}

// ──────────────────────────────────────────────
//  Root Colour Property
// ──────────────────────────────────────────────

TEST(RootColour, BlackAfterSingleInsert)
{
    IntTree tree;
    tree.add(1, "a");
    auto col = nc(tree, tree.root_node());
    EXPECT_EQ(col, Colour::Black);
}

TEST(RootColour, BlackAfterEveryInsert)
{
    IntIntTree tree;
    for (int k = 1; k < 20; ++k)
    {
        tree.add(k, k);
        auto col = nc(tree, tree.root_node());
        EXPECT_EQ(col, Colour::Black) << "Root not black after inserting " << k;
    }
}

TEST(RootColour, BlackAfterDeletes)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3, 7})
        tree.add(k, std::to_string(k));
    for (int k : {10, 5, 15})
    {
        tree.remove(k);
        if (tree.size() > 0)
        {
            auto col = nc(tree, tree.root_node());
            EXPECT_EQ(col, Colour::Black);
        }
    }
}

// ──────────────────────────────────────────────
//  No Red-Red Violation Property
// ──────────────────────────────────────────────

TEST(NoRedRedViolation, AfterSequentialInserts)
{
    IntIntTree tree;
    for (int k = 1; k <= 15; ++k)
    {
        tree.add(k, k);
        assert_no_red_red(tree);
    }
}

TEST(NoRedRedViolation, AfterDeletes)
{
    IntIntTree tree;
    for (int k = 1; k <= 15; ++k)
        tree.add(k, k);
    for (int k : {8, 4, 12, 2, 6, 10, 14})
    {
        tree.remove(k);
        assert_no_red_red(tree);
    }
}

// ──────────────────────────────────────────────
//  Parent Pointer Consistency
// ──────────────────────────────────────────────

TEST(ParentPointers, ConsistentAfterInserts)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3, 7, 12, 20})
        tree.add(k, std::to_string(k));
    assert_parent_pointers(tree);
}

TEST(ParentPointers, ConsistentAfterDeletes)
{
    IntTree tree;
    for (int k : {10, 5, 15, 3, 7, 12, 20})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    tree.remove(15);
    assert_parent_pointers(tree);
}

TEST(ParentPointers, RootParentIsNil)
{
    IntTree tree;
    tree.add(10, "ten");
    EXPECT_EQ(np(tree, tree.root_node()), tree.nil_node());
}

TEST(ParentPointers, ConsistentAfterRotations)
{
    IntIntTree tree;
    for (int k = 1; k <= 10; ++k)
        tree.add(k, k);
    assert_parent_pointers(tree);
}

// ──────────────────────────────────────────────
//  Stress / Scale
// ──────────────────────────────────────────────

TEST(Stress, ThousandAscendingInserts)
{
    IntIntTree tree;
    for (int k = 0; k < 1000; ++k)
        tree.add(k, k);
    EXPECT_EQ(tree.size(), 1000u);
    assert_rb_properties(tree);
}

TEST(Stress, Insert500DeleteEvens)
{
    IntIntTree tree;
    for (int k = 0; k < 500; ++k)
        tree.add(k, k);
    for (int k = 0; k < 500; k += 2)
        tree.remove(k);
    EXPECT_EQ(tree.size(), 250u);
    assert_rb_properties(tree);
    for (int k = 1; k < 500; k += 2)
        EXPECT_EQ(tree.find(k)->get(), k);
}

TEST(Stress, InsertThenDeleteAll1000)
{
    IntIntTree tree;
    for (int k = 0; k < 1000; ++k)
        tree.add(k, k);
    for (int k = 0; k < 1000; ++k)
        EXPECT_TRUE(tree.remove(k));
    EXPECT_EQ(tree.size(), 0u);
}

TEST(Stress, FindAllInLargeTree)
{
    IntIntTree tree;
    for (int k = 0; k < 500; ++k)
        tree.add(k, k * 2);
    for (int k = 0; k < 500; ++k)
        EXPECT_EQ(tree.find(k)->get(), k * 2);
    for (int k = 500; k < 600; ++k)
        EXPECT_FALSE(tree.find(k).has_value());
}

// ──────────────────────────────────────────────
//  Edge Cases
// ──────────────────────────────────────────────

TEST(EdgeCases, InsertZeroKey)
{
    IntTree tree;
    tree.add(0, "zero");
    EXPECT_EQ(tree.find(0)->get(), "zero");
    EXPECT_EQ(tree.size(), 1u);
}

TEST(EdgeCases, InsertNegativeKeys)
{
    IntTree tree;
    tree.add(-1, "neg1");
    tree.add(-100, "neg100");
    EXPECT_EQ(tree.find(-1)->get(), "neg1");
    EXPECT_EQ(tree.find(-100)->get(), "neg100");
    assert_rb_properties(tree);
}

TEST(EdgeCases, SingleElementDeleteLeavesEmptyTree)
{
    IntTree tree;
    tree.add(42, "x");
    tree.remove(42);
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_FALSE(tree.find(42).has_value());
    tree.add(99, "y");
    EXPECT_EQ(tree.find(99)->get(), "y");
}

TEST(EdgeCases, DeleteMinimumKey)
{
    IntTree tree;
    for (int k : {20, 10, 30, 5, 15})
        tree.add(k, std::to_string(k));
    tree.remove(5);
    EXPECT_FALSE(tree.find(5).has_value());
    assert_rb_properties(tree);
}

TEST(EdgeCases, DeleteMaximumKey)
{
    IntTree tree;
    for (int k : {20, 10, 30, 5, 15})
        tree.add(k, std::to_string(k));
    tree.remove(30);
    EXPECT_FALSE(tree.find(30).has_value());
    assert_rb_properties(tree);
}

TEST(EdgeCases, ManyDuplicatesNoSizeGrowth)
{
    IntTree tree;
    tree.add(42, "v1");
    for (int i = 0; i < 100; ++i)
        tree.add(42, "v" + std::to_string(i + 2));
    EXPECT_EQ(tree.size(), 1u);
    EXPECT_EQ(tree.find(42)->get(), "v101");
}

TEST(EdgeCases, DeleteAllThenReinsert)
{
    IntTree tree;
    for (int k = 1; k <= 5; ++k)
        tree.add(k, std::to_string(k));
    for (int k = 1; k <= 5; ++k)
        tree.remove(k);
    EXPECT_EQ(tree.size(), 0u);
    for (int k = 10; k < 15; ++k)
        tree.add(k, std::to_string(k));
    EXPECT_EQ(tree.size(), 5u);
    for (int k = 10; k < 15; ++k)
        EXPECT_EQ(tree.find(k)->get(), std::to_string(k));
    assert_rb_properties(tree);
}

TEST(EdgeCases, TreeHeightIsLogarithmic)
{
    IntIntTree tree;
    for (int k = 0; k < 1000; ++k)
        tree.add(k, k);

    auto nil = tree.nil_node();
    std::function<int(decltype(nil))> height = [&](decltype(nil) node) -> int
    {
        if (node == nil)
            return 0;
        auto l = IntIntTree::node_left(node);
        auto r = IntIntTree::node_right(node);
        return 1 + std::max(height(l), height(r));
    };

    int h = height(tree.root_node());
    double max_allowed = 2.0 * std::log2(1001.0) + 1.0;
    EXPECT_LE(h, static_cast<int>(max_allowed))
        << "Height " << h << " exceeds expected max " << max_allowed;
}

// ──────────────────────────────────────────────
//  Move Semantics
// ──────────────────────────────────────────────

TEST(MoveSemantics, MoveConstruct)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    IntTree moved(std::move(tree));
    EXPECT_EQ(moved.size(), 3u);
    EXPECT_EQ(moved.find(10)->get(), "10");
    assert_rb_properties(moved);
}

TEST(MoveSemantics, MoveAssign)
{
    IntTree tree;
    for (int k : {10, 5, 15})
        tree.add(k, std::to_string(k));
    IntTree other;
    other = std::move(tree);
    EXPECT_EQ(other.size(), 3u);
    EXPECT_EQ(other.find(5)->get(), "5");
    assert_rb_properties(other);
}

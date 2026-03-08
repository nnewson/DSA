#ifndef RED_BLACK_TREE_HPP
#define RED_BLACK_TREE_HPP

#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <stack>
#include <stdexcept>
#include <utility>
#include <vector>

namespace red_black_tree
{

enum class Colour
{
    Red,
    Black
};

/**
 * @brief A self-balancing binary search tree that maintains O(log n) lookup, insertion,
 *        and deletion by enforcing red-black colouring invariants.
 *
 * @tparam K Key type, must be comparable via the Compare functor.
 * @tparam V Value type stored alongside each key.
 * @tparam Compare Strict weak ordering functor for keys (defaults to std::less<K>).
 *
 * @details
 * Uses a sentinel nil node instead of nullptr to simplify boundary checks — every
 * leaf and the root's parent point to this single shared sentinel. The nil node is
 * always black, which allows rotation and fixup code to read colour fields without
 * null-checking.
 */
template <typename K, typename V, typename Compare = std::less<K>>
class RedBlackTree
{
    struct Node
    {
        K key;
        V value;
        Colour colour = Colour::Red;
        Node* left = nullptr;
        Node* right = nullptr;
        Node* parent = nullptr; // non-owning back-pointer; ownership flows parent -> child
    };

public:
    RedBlackTree()
        : nil_{make_nil()},
          root_{nil_},
          size_{0}
    {
    }

    ~RedBlackTree()
    {
        destroy(root_);
        delete nil_;
    }

    RedBlackTree(const RedBlackTree&) = delete;
    RedBlackTree& operator=(const RedBlackTree&) = delete;

    RedBlackTree(RedBlackTree&& other) noexcept
        : nil_{other.nil_},
          root_{other.root_},
          size_{other.size_},
          comp_{std::move(other.comp_)}
    {
        other.nil_ = nullptr;
        other.root_ = nullptr;
        other.size_ = 0;
    }

    RedBlackTree& operator=(RedBlackTree&& other) noexcept
    {
        if (this != &other)
        {
            destroy(root_);
            delete nil_;
            nil_ = other.nil_;
            root_ = other.root_;
            size_ = other.size_;
            comp_ = std::move(other.comp_);
            other.nil_ = nullptr;
            other.root_ = nullptr;
            other.size_ = 0;
        }

        return *this;
    }

    [[nodiscard]]
    std::size_t size() const noexcept
    {
        return size_;
    }

    [[nodiscard]]
    bool empty() const noexcept
    {
        return size_ == 0;
    }

    [[nodiscard]]
    bool contains(const K& key) const
    {
        auto [node, found] = find_node_or_parent(key);
        return found;
    }

    [[nodiscard]]
    const V& at(const K& key) const
    {
        auto [node, found] = find_node_or_parent(key);
        if (!found)
        {
            throw std::out_of_range("key not found");
        }

        return node->value;
    }

    V& operator[](const K& key)
    {
        auto [node, found] = find_node_or_parent(key);
        if (!found)
        {
            throw std::out_of_range("key not found");
        }

        return node->value;
    }

    const V& operator[](const K& key) const
    {
        return at(key);
    }

    /**
     * @brief Inserts a key-value pair into the red-black tree.
     *
     * @param key The key to insert. Must be comparable and unique within the tree.
     * @param value The value associated with the key.
     *
     * @note If the key already exists, the behavior depends on the implementation
     *       (e.g., update the existing value or ignore the insertion).
     */
    void add(const K& key, const V& value)
    {
        auto [parent_node, found] = find_node_or_parent(key);
        if (found)
        {
            parent_node->value = value;
            return;
        }

        ++size_;
        // New nodes start red with both children pointing to nil_
        auto* new_node = new Node{key, value, Colour::Red, nil_, nil_, nil_};

        if (parent_node == nil_)
        {
            // Tree was empty — new node becomes the root
            root_ = new_node;
        }
        else if (comp_(key, parent_node->key))
        {
            parent_node->left = new_node;
            new_node->parent = parent_node;
        }
        else
        {
            parent_node->right = new_node;
            new_node->parent = parent_node;
        }

        // A red node under a red parent violates RB properties — fix up the tree
        insert_fixup(new_node);
    }

    /**
     * @brief Finds the value associated with the specified key.
     *
     * Performs a lookup using the tree comparator and returns a non-owning
     * reference to the stored value when the key exists.
     *
     * @param key The key to search for.
     *
     * @return A populated std::optional containing a const reference to the value
     *         if the key is present; otherwise std::nullopt.
     */
    [[nodiscard]]
    std::optional<std::reference_wrapper<const V>> find(const K& key) const
    {
        auto [node, found] = find_node_or_parent(key);
        if (!found)
        {
            return std::nullopt;
        }

        return std::cref(node->value);
    }

    /**
     * @brief Removes a node with the specified key from the red-black tree.
     *
     * Performs a standard BST deletion followed by red-black tree fixup if necessary.
     * Handles three cases: node with no left child, node with no right child, and node
     * with two children (using in-order successor replacement).
     *
     * @param key The key of the node to remove
     *
     * @return true if a node with the specified key was found and removed,
     *         false if no node with the specified key exists in the tree
     *
     * @details
     * The function maintains red-black tree properties by:
     * 1. Tracking the color of the node being removed or its successor
     * 2. Identifying which node will replace the deleted node (fixup_node)
     * 3. Calling delete_fixup() if a black node was removed, to restore
     *    the black-height property
     *
     * Time Complexity: O(log n) where n is the number of nodes
     * Space Complexity: O(1)
     */
    bool remove(const K& key)
    {
        auto [node_to_delete, found] = find_node_or_parent(key);
        if (!found)
        {
            return false;
        }

        // Track the colour of the node physically removed from the tree.
        // If it was black, deletion reduces black-height and requires fixup.
        auto original_colour = node_to_delete->colour;
        Node* fixup_node; // the node that takes the deleted node's position

        if (node_to_delete->left == nil_)
        {
            // Only a right child (or no children) — splice in the right subtree
            fixup_node = node_to_delete->right;
            transplant(node_to_delete, node_to_delete->right);
        }
        else if (node_to_delete->right == nil_)
        {
            // Only a left child — splice in the left subtree
            fixup_node = node_to_delete->left;
            transplant(node_to_delete, node_to_delete->left);
        }
        else
        {
            // Two children — replace with the in-order successor (smallest node
            // in the right subtree). The successor has no left child by definition.
            auto* successor = minimum(node_to_delete->right);
            original_colour = successor->colour;
            fixup_node = successor->right;

            if (successor->parent == node_to_delete)
            {
                // Successor is the direct right child — fixup_node's parent is
                // already correct after transplant, but we must set it explicitly
                // because fixup_node may be nil_ (whose parent is self-referential).
                fixup_node->parent = successor;
            }
            else
            {
                // Detach successor from its current position, then adopt
                // node_to_delete's right subtree
                transplant(successor, successor->right);
                successor->right = node_to_delete->right;
                successor->right->parent = successor;
            }

            // Move successor into node_to_delete's position and adopt its left subtree
            transplant(node_to_delete, successor);
            successor->left = node_to_delete->left;
            successor->left->parent = successor;
            successor->colour = node_to_delete->colour;
        }

        if (original_colour == Colour::Black)
        {
            // Removing a black node broke the equal-black-height invariant
            delete_fixup(fixup_node);
        }

        delete node_to_delete;
        --size_;
        return true;
    }

    /**
     * @brief Stack-based in-order iterator over the tree's key-value pairs.
     *
     * @details Uses an explicit stack to simulate the recursive in-order traversal.
     *          On construction, pushes all left-spine nodes from the root. Each
     *          advance pops the top node (the next smallest) and pushes the left
     *          spine of its right subtree. Reaching an empty stack signals end().
     */
    class Iterator
    {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::pair<const K&, const V&>;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = value_type;

        Iterator() = default;

        reference operator*() const
        {
            return {current_->key, current_->value};
        }

        Iterator& operator++()
        {
            advance();
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator tmp = *this;
            advance();
            return tmp;
        }

        friend bool operator==(const Iterator& a, const Iterator& b)
        {
            return a.current_ == b.current_;
        }

        friend bool operator!=(const Iterator& a, const Iterator& b)
        {
            return !(a == b);
        }

    private:
        friend class RedBlackTree;

        Iterator(Node* root, Node* nil)
            : nil_{nil}
        {
            push_left(root);
            advance_to_next();
        }

        // Sentinel constructor for end()
        explicit Iterator(Node* nil)
            : nil_{nil},
              current_{nil}
        {
        }

        void push_left(Node* node)
        {
            while (node != nil_)
            {
                stack_.push(node);
                node = node->left;
            }
        }

        void advance_to_next()
        {
            if (stack_.empty())
            {
                current_ = nil_;
                return;
            }

            current_ = stack_.top();
            stack_.pop();
            push_left(current_->right);
        }

        void advance()
        {
            advance_to_next();
        }

        Node* nil_ = nullptr;
        Node* current_ = nullptr;
        std::stack<Node*> stack_;
    };

    [[nodiscard]]
    Iterator begin() const
    {
        return Iterator{root_, nil_};
    }

    [[nodiscard]]
    Iterator end() const
    {
        return Iterator{nil_};
    }

    // Collect in-order as vector of pairs (useful for testing)
    [[nodiscard]]
    std::vector<std::pair<K, V>> collect_inorder() const
    {
        std::vector<std::pair<K, V>> result;
        result.reserve(size_);

        for (auto [k, v] : *this)
        {
            result.emplace_back(k, v);
        }

        return result;
    }

    // Expose internals for testing RB properties
    struct NodeView
    {
        const K* key;
        Colour colour;
        const NodeView* left;
        const NodeView* right;
        const NodeView* parent;
        bool is_nil;
    };

    // Access to root and nil for property validation in tests
    [[nodiscard]]
    const Node* root_node() const
    {
        return root_;
    }

    [[nodiscard]]
    const Node* nil_node() const
    {
        return nil_;
    }

    // Expose Node internals for test helpers
    static const K& node_key(const Node* n)
    {
        return n->key;
    }

    static const V& node_value(const Node* n)
    {
        return n->value;
    }

    static Colour node_colour(const Node* n)
    {
        return n->colour;
    }

    static const Node* node_left(const Node* n)
    {
        return n->left;
    }

    static const Node* node_right(const Node* n)
    {
        return n->right;
    }

    static const Node* node_parent(const Node* n)
    {
        return n->parent;
    }

private:
    /**
     * @brief Creates the sentinel nil node used as a universal leaf/boundary marker.
     *
     * @details The nil node's left, right, and parent pointers all reference itself,
     *          forming a self-referential cycle. This allows code to unconditionally
     *          dereference any node pointer (e.g., sibling->left->colour) without
     *          null checks, since traversal past a leaf always lands on nil.
     */
    static Node* make_nil()
    {
        auto* nil = new Node{};
        nil->colour = Colour::Black;
        nil->left = nil;
        nil->right = nil;
        nil->parent = nil;
        return nil;
    }

    /**
     * @brief Recursively deallocates all nodes in the subtree rooted at node.
     * @note Stops at nil_ or nullptr to avoid destroying the sentinel.
     *       The nil_ node is deleted separately in the destructor.
     */
    void destroy(Node* node)
    {
        if (node == nil_ || node == nullptr)
        {
            return;
        }

        destroy(node->left);
        destroy(node->right);
        delete node;
    }

    /**
     * @brief Searches the tree for a node matching the given key.
     *
     * @return A pair of (node_pointer, found). When found is true, node_pointer is
     *         the matching node. When found is false, node_pointer is where a new
     *         node with this key would be inserted as a child (the would-be parent),
     *         or nil_ if the tree is empty. This dual purpose avoids redundant
     *         traversals in add() and remove().
     */
    [[nodiscard]]
    std::pair<Node*, bool> find_node_or_parent(const K& key) const
    {
        Node* current = root_;
        Node* parent = nil_;

        while (current != nil_)
        {
            parent = current;
            if (comp_(key, current->key))
            {
                current = current->left;
            }
            else if (comp_(current->key, key))
            {
                current = current->right;
            }
            else
            {
                return {current, true};
            }
        }

        return {parent, false};
    }

    /** @brief Returns the leftmost (smallest) node in the subtree rooted at node. */
    Node* minimum(Node* node) const
    {
        while (node->left != nil_)
        {
            node = node->left;
        }
        return node;
    }

    /**
     * @brief Replaces the subtree rooted at 'from' with the subtree rooted at 'to'.
     *
     * @details Updates from's parent to point to 'to' instead, and sets to's parent
     *          accordingly. Does not update to's children — the caller is responsible
     *          for linking any remaining subtree references (see remove()).
     */
    void transplant(Node* from, Node* to)
    {
        if (from->parent == nil_)
        {
            root_ = to;
        }
        else if (from == from->parent->left)
        {
            from->parent->left = to;
        }
        else
        {
            from->parent->right = to;
        }
        to->parent = from->parent;
    }

    /**
     * @brief Performs a left rotation around the given node.
     *
     * @details Pivots node down and to the left, promoting its right child upward.
     *          Before:        After:
     *            node           R
     *           /    \        /   \
     *          A      R     node   C
     *               /   \  /    \
     *              B     C A     B
     *
     *          Preserves the BST ordering invariant and updates all parent pointers.
     */
    void rotate_left(Node* node)
    {
        Node* right_child = node->right;
        assert(right_child != nil_);

        node->right = right_child->left;
        if (right_child->left != nil_)
        {
            right_child->left->parent = node;
        }

        right_child->parent = node->parent;

        if (node->parent == nil_)
        {
            root_ = right_child;
        }
        else if (node == node->parent->left)
        {
            node->parent->left = right_child;
        }
        else
        {
            node->parent->right = right_child;
        }

        right_child->left = node;
        node->parent = right_child;
    }

    /**
     * @brief Performs a right rotation around the given node (mirror of rotate_left).
     *
     * @details Pivots node down and to the right, promoting its left child upward.
     *          Before:        After:
     *            node           L
     *           /    \        /   \
     *          L      C     A    node
     *        /   \              /    \
     *       A     B            B      C
     */
    void rotate_right(Node* node)
    {
        Node* left_child = node->left;
        assert(left_child != nil_);

        node->left = left_child->right;
        if (left_child->right != nil_)
        {
            left_child->right->parent = node;
        }

        left_child->parent = node->parent;

        if (node->parent == nil_)
        {
            root_ = left_child;
        }
        else if (node == node->parent->right)
        {
            node->parent->right = left_child;
        }
        else
        {
            node->parent->left = left_child;
        }

        left_child->right = node;
        node->parent = left_child;
    }

    /**
     * @brief Restores red-black properties after insertion.
     *
     * @details A newly inserted node is always red, which may violate the rule that
     *          a red node cannot have a red parent. This method walks up the tree
     *          recolouring and rotating until the violation is resolved.
     *
     *          Three cases are handled (with symmetric mirrors for left/right):
     *            Case 1 — Uncle is red: recolour parent, uncle, and grandparent;
     *                     move the violation up two levels and repeat.
     *            Case 2 — Uncle is black, node is an inner child (zig-zag): rotate
     *                     node's parent to convert into Case 3.
     *            Case 3 — Uncle is black, node is an outer child (zig-zig): recolour
     *                     and rotate grandparent. This is the terminal case.
     */
    void insert_fixup(Node* node)
    {
        while (node->parent != nil_ && node->parent->colour == Colour::Red)
        {
            // Determine which side of the grandparent the parent sits on.
            // The logic below handles parent == grandparent->left;
            // the else branch is the symmetric mirror.
            if (node->parent == node->parent->parent->left)
            {
                Node* uncle = node->parent->parent->right;

                if (uncle != nil_ && uncle->colour == Colour::Red)
                {
                    // Case 1: uncle is red — push blackness down from grandparent
                    node->parent->colour = Colour::Black;
                    uncle->colour = Colour::Black;
                    node->parent->parent->colour = Colour::Red;
                    node = node->parent->parent;
                }
                else
                {
                    if (node == node->parent->right)
                    {
                        // Case 2: uncle black, node is right child (zig-zag)
                        // Rotate to align node as left child, converting to Case 3
                        node = node->parent;
                        rotate_left(node);
                    }

                    // Case 3: uncle black, node is left child (zig-zig)
                    node->parent->colour = Colour::Black;
                    node->parent->parent->colour = Colour::Red;
                    rotate_right(node->parent->parent);
                }
            }
            else // Mirror: parent is right child of grandparent
            {
                Node* uncle = node->parent->parent->left;

                if (uncle != nil_ && uncle->colour == Colour::Red)
                {
                    // Case 1 (mirror)
                    node->parent->colour = Colour::Black;
                    uncle->colour = Colour::Black;
                    node->parent->parent->colour = Colour::Red;
                    node = node->parent->parent;
                }
                else
                {
                    if (node == node->parent->left)
                    {
                        // Case 2 (mirror): rotate to convert to Case 3
                        node = node->parent;
                        rotate_right(node);
                    }

                    // Case 3 (mirror)
                    node->parent->colour = Colour::Black;
                    node->parent->parent->colour = Colour::Red;
                    rotate_left(node->parent->parent);
                }
            }
        }

        // The root must always be black (insertion may have coloured it red in Case 1)
        root_->colour = Colour::Black;
    }

    /**
     * @brief Restores red-black properties after deletion of a black node.
     *
     * @param node The replacement node that now occupies the deleted node's position.
     *             This node carries an "extra black" that must be resolved.
     *
     * @details Removing a black node reduces the black-height on one path, violating
     *          the invariant that all root-to-leaf paths have equal black-height.
     *          The algorithm treats 'node' as "doubly black" and resolves it by
     *          pushing the extra black up the tree or absorbing it via rotations.
     *
     *          Four cases are handled (with symmetric mirrors for left/right):
     *            Case 1 — Sibling is red: rotate parent to make sibling black,
     *                     converting to Cases 2, 3, or 4.
     *            Case 2 — Sibling is black with two black children: remove one black
     *                     from both node and sibling (recolour sibling red), push the
     *                     extra black up to the parent and repeat.
     *            Case 3 — Sibling is black, far child is black, near child is red:
     *                     rotate sibling to make the far child red, converting to Case 4.
     *            Case 4 — Sibling is black, far child is red: rotate parent and
     *                     recolour to absorb the extra black. This is the terminal case.
     */
    void delete_fixup(Node* node)
    {
        while (node != root_ && node->colour == Colour::Black)
        {
            // Handle node being a left child; the else branch is the symmetric mirror.
            if (node == node->parent->left)
            {
                Node* sibling = node->parent->right;
                if (sibling->colour == Colour::Red)
                {
                    // Case 1: sibling is red — rotate to get a black sibling
                    sibling->colour = Colour::Black;
                    node->parent->colour = Colour::Red;
                    rotate_left(node->parent);
                    sibling = node->parent->right;
                }

                if (sibling->left->colour == Colour::Black &&
                    sibling->right->colour == Colour::Black)
                {
                    // Case 2: sibling black with two black children —
                    // pull a black off both node and sibling, push extra black to parent
                    sibling->colour = Colour::Red;
                    node = node->parent;
                }
                else
                {
                    if (sibling->right->colour == Colour::Black)
                    {
                        // Case 3: near child (left) is red, far child (right) is black —
                        // rotate sibling right to set up Case 4
                        sibling->left->colour = Colour::Black;
                        sibling->colour = Colour::Red;
                        rotate_right(sibling);
                        sibling = node->parent->right;
                    }

                    // Case 4: far child (right) is red — absorb the extra black
                    sibling->colour = node->parent->colour;
                    node->parent->colour = Colour::Black;
                    sibling->right->colour = Colour::Black;
                    rotate_left(node->parent);
                    node = root_; // terminates the loop
                }
            }
            else // Mirror: node is a right child
            {
                Node* sibling = node->parent->left;

                if (sibling->colour == Colour::Red)
                {
                    // Case 1 (mirror)
                    sibling->colour = Colour::Black;
                    node->parent->colour = Colour::Red;
                    rotate_right(node->parent);
                    sibling = node->parent->left;
                }
                if (sibling->right->colour == Colour::Black &&
                    sibling->left->colour == Colour::Black)
                {
                    // Case 2 (mirror)
                    sibling->colour = Colour::Red;
                    node = node->parent;
                }
                else
                {
                    if (sibling->left->colour == Colour::Black)
                    {
                        // Case 3 (mirror)
                        sibling->right->colour = Colour::Black;
                        sibling->colour = Colour::Red;
                        rotate_left(sibling);
                        sibling = node->parent->left;
                    }

                    // Case 4 (mirror)
                    sibling->colour = node->parent->colour;
                    node->parent->colour = Colour::Black;
                    sibling->left->colour = Colour::Black;
                    rotate_right(node->parent);
                    node = root_;
                }
            }
        }

        // Resolve any remaining extra black by simply colouring the node black
        node->colour = Colour::Black;
    }

    Node* nil_;
    Node* root_;
    std::size_t size_;
    [[no_unique_address]] Compare comp_;
};

} // namespace red_black_tree

#endif // RED_BLACK_TREE_HPP

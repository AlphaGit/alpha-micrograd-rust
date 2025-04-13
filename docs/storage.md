Thinking about tree storage in an array

Usually they will be stored root-first, so a tree like:

  A
B   C

Will be stored like:

[A, B, C]

For incomplete trees:

    A
  B   C
D    F  G

(E is missing under the right side of B), we'd store it like this:

[A, B, C, D, None, F, G]

For leaf-to-root traversal of the tree, we'd have to traverse the array from end to beggining.

Now, inserting a new root is complex because we'd have to move the whole array.

Both inefficiencies can be solved if we store the arrays inverted, assuming we won't add new leaves at the bottom.

  A
B   C

-> 

[C, B, A]

     A
  B      C
D   x  F    G

[G, F, None, D, C, B, A]

Let's try adding roots to these:

Adding a single node Z to the top:

                Z
  A          A     x
B   C  ->  B   C  x  x

[C, B, A] -> [x, x, C, B, x, A, Z]

Not only we had to move the array, we also have to add elements in the middle. This happens because adding a new root can carry a whole new subtree with it. This is actually not just adding a single node, but actually merging trees.

Assuming it's okay to sacrifice performance to do it, adding a new element for a tree would be (using regular ordering with root-first):

- inputs: subtreeA, subtreeB (root-first order)
- select the new root and add it to the new tree
- select 1 element from subtreeA add add it to the new tree (left)
- select 1 new element from subtreeA and add it to the new tree (right)
- select +2 elements from subtreeA, add them to the new tree (level 2)
- select +2 elements from subtreeB, add them to the new tree (level 2)
- repeat until both trees are finished
    - if a tree if finished first, insert nones instead

To achieve tree merging in a reverse order, we can either do that and reverse the result, OR merge them in reverse:

- inputs: subtreeA, subtreeB (root-last order)
- levelsA = log2(len(subtreeA)), levelsB = log2(len(subtreeB))
- lastLevel = max(levelsA, levelsB)
- take N elements from the end of subtreeB and add them to the new tree
    - N: how many elements are in lastLevel (lastLevel**2)
    - if a tree does not have elements in that level, replace with that amount of Nones
- decrease the level being processed
- repeat until all levels have been processed
- add the new root to the end of the array

For our case, this would result in a lot of Nones because our trees are highly unbalanced: when a new root is added, the other subtree likely has  one other leaf next to it.
    - is this true??
        - no, it's not:
            - merging complex operations with a single operation (sum/mul)
            - multiple neurones being sent to a single activation function
                - but is this how they're connected at the low level?

---

Now that the algorithm is working, let's define the structures that we'd use to implement this:

If the vector array was contained by the Expr (value) structures, only one of them would be able to mutate them. Maybe this is good enough, as these kind of operations should happen from the root only.

Another alternative is to have a single aggregating entity that worked like a container for all the expressions. This could be similar to tensorflow's Session() object. If we went this route, then the operations would not happen against the expressions directly (exprA + exprB) but rather should be operations on the session itself -- but this would give the wrong sensation that the operation is happening on the whole tree.

```rust
let session = Session::new();
// this could initialize the session with an empty tree

session.add_root(expr);
// doing this would define the root... but what would it do if called repeatedly?

session.add_element(expr);
// this could define the root... but otherwise it would add an orphan element? Might not be a good idea to have disjointed trees.

session.set_tree(expr);
// this could either set the root or a whole tree, replacing the existing one, but the naming is weird, as tree is a leaky abstraction here

session.set_contents(expr);
// this one feels better

// at this point, the session itself could behave like an expression, with operations adding and replacing the root

let b = session + expr;
// this feels like a natural interface, but at this point, it's the same as the Expr behaviour we have today

let expr = Expr::new(1.0); // generates a tree with a single element
let expr2 = Expr::new(2.0); // generates a tree with a single element
let exprSum = expr + expr2; // this would merge the trees and generate a new one
// but merging the trees requires having references to each vector element
// if these are owned by the vector, both expr and expr2 should be owned under exprSum (this is feasible).
// if these are just referenced by the vector, all of exprSum, expr and expr2 would keep their own prolongued existence
// which sounds very useful but what would it mean for these elements to be re-processed? each individual tree would end up mutating them and the results would be... weird.
```

Since owning sounds like a better idea, the result of an operation should get ownership of the other operands and merge their trees:

- any unary operation: add root, add existing subtree to left side
    - tanh 
    - relu
    - exp
    - pow
    - log
    - neg
- recalculate: go from left to right (leaves to root) and calculate new values
    - calculate a value:
        - get child 1 (math operation, then indexing)
        - get child 2 (math operation, then indexing)
        - calculate operation and store in current index
    - used in training
    - used in inference
- learn: (backpropagation) go from right to left (root to leaves) and calculate new values for gradients
    - get current gradiend
    - get child 1 (math operation, then indexing)
    - get child 2  (math operation, then indexing)
    - set new gradient value to child 1
    - set new gradient value to child 2*(1)
    - used in training only
- find: find an element in the array
    - might be improved as a hashmap of strings --> indexes
- find_mut: same
- parametr_count: count non-None vector elements
    - can be pre-calculated as the tree will not change frequently
    - however, this operation would not be used very frequently
    - but the drawback is a single u32 value
- print_tree: go from right to left (root to leaves) and print the node information
- any binary operation: merge two trees (algorithm above), add a new root
    - add
    - mul
    - sub
    - div
- sum: (multiple disjointed values)
    - doing multiple adds would be very inefficient
    - but having more than 2 child nodes would break the binary tree struture
        - we need this for vector indexing
    - maybe we can accomodate all of these into a right tree-structure structure
        - but maybe others have both children too
    - we have these combinations of possible parameters
        - leaf + leaf --> can right align to the tree
        - leaf + unary --> can right align to the tree (unary's child will be left-aligned)
        - left + binary --> regular add (new root, children at both sides)
        - unary + leaf --> same as leaf + unary
        - unary + unary --> can right align
        - unary + binary --> regular add (new root, children at both sides)
        - binary + leaf --> regular add (new root, children at both sides)
        - binary + unary --> regular add (new root, children at both sides)
        - binary + binary --> regular add (new root, children at both sides)
    - it seems we need to fallback to the regular adding algorithm whenever a binary value is involved
    - because addition is commutative, we can align all leaf + unary in a single branch and then fallback to regular addition with the rest of the binaries


*(1): This might incur into a problem of double borrowing but we can get around it by using a single mutable borrow at a time:

```rust
let mut v = vec![1, 2, 3, 4, 5];

{
    let mut a = v.get_mut(1).unwrap();
    dbg!(a);
}

{
    let mut b = v.get_mut(2).unwrap();
    dbg!(b);
}
```

---

Structures:

```rust
struct Expr {
    tree: Vec<ExprNode>
}

struct ExprNode {
    operation: Operation,
    result: f64,
    is_learnable: bool,
    grad: f64
}
```

---

Adding a new root to a tree, moving the subtree to the left side:

Example: (index, node)

      0
      A
  1       2
  B       C
3    4  5   6
D    E  F   G

Adding Z as a root would now become (x: None)

              0
              Z
        1           2
        A           x
    3     4      5     6
    B     C      x     x
7    8  9   10 11 12 13 14
D    E  F    G  x  x  x  x

(premature optimization: we can save storing 11-14)

The values became:
- new root --> 0
- 0 (level 1) --> 1 (+1)
- 1 (level 2) --> 3 (+2)
- 2 (level 2) --> 4 (+2)
- 3 (level 3) --> 7 (+4)
- 4 (level 3) --> 8 (+4)
- 5 (level 3) --> 9 (+4)
- 6 (level 3) --> 10 (+4)

If there was a child to D, it'd be 7 (level 4) --> 15 (+8)

So the repositioning algorithm is:

- get level number from index
    - root: level 0
    - floor(log(i)/log(2))
    - optimization: count position of last on bit (right to left)
- add 2**(level-1) to index
    - optimization: ?

index mapping in binary:

- newr + 0000 -> 0000
- 0000 + 0001 -> 0001
- 0001 + 0010 -> 0011
- 0010 + 0010 -> 0100
- 0011 + 0100 -> 0111
- 0100 + 0100 -> 1000
- 0101 + 0100 -> 1001
- 0110 + 0100 -> 1010
- 0111 + 1000 -> 1111

----

Doing the same exercise but with reversed-order trees:

Example: (index, node)

      6
      A
  5       4
  B       C
3    2  1   0
D    E  F   G

Adding Z as a root would now become (x: None)

             14
              Z
        13         12
        A           x
    11   10      9     8
    B     C      x     x
7    6  5    4  3  2  1  0
D    E  F    G  x  x  x  x

Let's analyze the pattern for index mapping when adding a new root in reverse-order trees:

- old index 6 (A) -> new index 13 (A)
- old index 5 (B) -> new index 11 (B)
- old index 4 (C) -> new index 10 (C)
- old index 3 (D) -> new index 7 (D)
- old index 2 (E) -> new index 6 (E)
- old index 1 (F) -> new index 5 (F)
- old index 0 (G) -> new index 4 (G)

Smaller example:

  2
  A
1   0
B   C

Adding root:

    6
    Z
  5    4
  A    x
3  2  1  0
B  C  x  x

- old index 2 (A) -> new index 6 (A)
- old index 1 (B) -> new index 3 (B)
- old index 0 (C) -> new index 2 (C)

---

In a regular (root-first) array representation of a tree:

- for a node in ith location
    - left child is at: 2i + 1
    - right child is at: 2i + 2
    - parent is at floor((i-1)/2)

if the tree is reversed, then every i corresponds to the other side (len-1-i)

    - left child is at: len-1-(2(len-1-i)+1)
        = len-1-2(len-1-i)-1
        = len-1-2len+2+2i-1
        = -len-1+2+2i-1
        = -len+2i-1+2-1
        = -len+2i
    - right child is at: len-1-(2(len-1-i)+2)
        = len-1-2(len-1-i)-2
        = len-1-2len+2+2i-2
        = -len-1+2+2i-2
        = -len-1+2i
        = -len+2i-1
    - parent is at len-1-floor((len-1-i-1)/2)
        = len-1-floor((len-i-2)/2)

        consider that: floor(x) + k = floor(x+k)
        https://math.stackexchange.com/questions/1086156/distribution-and-other-rules-for-floor-and-ceiling

        = -floor((len-i-2)/2 + len-1)
        = -floor((len-i-2)/2 + (2len-2)/2)
        = -floor(((len-i-2)+(2len-2)))/2
        = -floor((3len-i-4)/2)

testing this out:

    6
    Z
  5    4
  A    X
3  2  1  0
B  C  T  U

left child of Z: -len+2i
    = -7+2*6
    = -7+12
    = 5 (correct)
right child of Z: -len+2i-1
    = -7+2*6-1
    = -7+12-1
    = 4 (correct)

left child of A: -len+2i
    = -7+2*5
    = -7+10
    = 3 (correct)
right child of A: -len+2i-1
    = ...
    = 2 (correct)

left child of X: -len+2i
    = -7+2*4
    = -7+8
    = 1 (correct)
right child of X: -len+2i-1
    = ...
    = 0 (correct)

Another tree:

    2
    A
1       0
B       C

left child of A: -len+2i
    = -3+2*2
    = -3+4
    = 1 (correct)
right child of A: -len+2i-1
    = ...
    = 0 (correct)
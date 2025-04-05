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


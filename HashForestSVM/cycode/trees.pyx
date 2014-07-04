
def say_hello_to(name):
    print("Hello %s!!" % name)

cdef class Node:
    cdef int col
    cdef double cut
    cdef Node left
    cdef Node right

    def __init__(self, int col, double cut):
        self.col = col
        self.cut = cut
    def __str__(self):
        return "feat=%d @ %1.3f" % (self.col, self.cut)
    
    property left:
        def __get__(self):
            return self.left
        def __set__(self, Node value):
            self.left = value

    property right:
        def __get__(self):
            return self.right
        def __set__(self, Node value):
            self.right = value

cpdef Node aTree():
    cdef Node root, child1
    root = Node(1, 10.1)
    child1 = Node(5, 5.75)
    root.left = child1
    return root


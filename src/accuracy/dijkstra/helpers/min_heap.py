import heapq


class Min_heap:
    def __init__(self):
        self.heap = []

    def add(self, elt) -> None:
        """
        Pushes an element to the heap.

        Args:
            elt: element to remove
        """
        heapq.heappush(self.heap, elt)

    def pop(self) -> None:
        """
        Removes the smallest element in the heap.
        """
        heapq.heappop(self.heap)

    def top(self):
        """
        Returns:
            the smallest element in the heap.
        """
        return self.heap[0]

    def size(self) -> int:
        """
        Returns:
            amount of elements in the heap.
        """
        return len(self.heap)

    def __iter__(self):
        for elt in self.heap:
            yield elt

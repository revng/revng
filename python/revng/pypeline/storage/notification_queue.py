#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from asyncio import Queue
from weakref import ReferenceType, ref


class MultiQueue[T]:
    """
    Implements multiple queues where a single message can be dispatched to all
    of them.
    Each consumer uses `get_queue` to get a queue instance where messages are
    sent to while each producer uses send.
    This assumes it's used in an async context, so it's not thread-safe.
    """

    def __init__(self):
        # List of weak references to the queues that were emitted via
        # `get_queue`, they need to be weak to allow the GC to get rid of them
        # if the user of the queue stops using them.
        self.queues: list[ReferenceType[Queue[T]]] = []

    def get_queue(self) -> Queue[T]:
        """Get a new queue where messages will be delivered to"""

        # Create a new queue and save a weak reference for us
        result: Queue[T] = Queue()
        self.queues.append(ref(result))
        return result

    def send(self, message: T):
        """Send a message to all the active queues"""

        # Copy the queue and clear it, a new list will be made based on which
        # queues have not been GC'ed in the meantime
        queues_copy = self.queues.copy()
        self.queues.clear()
        for queue_ref in queues_copy:
            queue = queue_ref()
            # If queue is None here it means that the queue was garbage
            # collected in the meantime, skip it and don't add it back in the
            # list
            if queue is not None:
                self.queues.append(queue_ref)
                queue.put_nowait(message)


LOCAL_QUEUE: MultiQueue[bytes] = MultiQueue()

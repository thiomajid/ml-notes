## Why choose databases over files ?
When writing to a file, many things have to considered:
- Is the writing process *atomic* ?
- What if I need to read the file *concurrently* ?
- Is the data really persisted to disk ? Maybe it's still in OS's page cache after the `write` operation and suddenly the OS crashes.

We can ensure a kind of atomicity by creating a **temporary** file that can later on be renamed. But this doesn't still control when is the data persisted to disk. To do so, one can *flush* the data to the disk before renaming the file. On a Linux system, this is done using the `fsync` syscall.

Another approach would be *append-only logs* but how could deleted data be handled given that it can't grow forever ? Also this approach leverage indexes for efficient data querying.

## Indexing
Almost all queries can be broken down into three types of disk operations:
1. Scan the **whole data set**. (No index is used). 
2. *Point query*: Query the index by a specific key. 
3. *Range query*: Query the index by a range. (The index is sorted).

>[!info]
>A memory page is a fixed-length contiguous block of virtual memory, used as the basic unit of memory allocation and transfer between main memory (RAM) and secondary storage (like a hard drive or SSD)

To store data in a convenient way that will allow fast queries, ordering and sorting many data structures can be considered: *hashtables, B-tree and LSM-tree*.
Given that we need most of the time sorting and ordering, hashtables are less relevant, that's why most of implementations relies on tree data structures.
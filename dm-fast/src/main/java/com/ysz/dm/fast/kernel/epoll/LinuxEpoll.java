package com.ysz.dm.fast.kernel.epoll;

import java.util.LinkedList;
import java.util.TreeMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * 学习规划:
 * <p>
 * V1 版本功能实现:
 * - 大体的模型:
 * - Epoll 的创建和2个核心模型 : 红黑树 和 双端链表
 * - Epoll
 * <p>
 * V2 Epoll 基于 MmapBuffer 的持久化操作:
 * - Java Mmaped 操作的 bug
 */
public class LinuxEpoll {

    private ReentrantReadWriteLock lock;

    private ReentrantReadWriteLock.ReadLock readLock;

    private ReentrantReadWriteLock.WriteLock writeLock;

    private TreeMap<Long, LinuxEpollRbItem> rbr;

    private LinkedList<LinuxEpollRbItem> rdllist;


    public LinuxEpoll() {
        this.lock = new ReentrantReadWriteLock();
        this.readLock = lock.readLock();
        this.writeLock = lock.writeLock();

        this.rbr = new TreeMap<>();
        this.rdllist = new LinkedList<>();
    }


    /**
     * @return null 表示插入失败
     */
    public LinuxEpollRbItem epollInsert(Long id) {
        LinuxEpollRbItem item = new LinuxEpollRbItem(id, this);

        readLock.lock();
        try {
            if (rbr.containsKey(item.getId())) return null;
        } finally {
            readLock.unlock();
        }


        writeLock.lock();
        try {
            /*ABA*/
            if (rbr.containsKey(item.getId())) return null;
            this.rbr.put(item.getId(), item);
            return item;
        } finally {
            writeLock.unlock();
        }
    }


}

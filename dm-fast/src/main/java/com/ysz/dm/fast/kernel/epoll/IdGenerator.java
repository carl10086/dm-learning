package com.ysz.dm.fast.kernel.epoll;

import java.util.concurrent.atomic.AtomicLong;

public class IdGenerator {

    private static AtomicLong atomicLong = new AtomicLong(0L);

    public static long nextId() {
        return atomicLong.incrementAndGet();
    }
}

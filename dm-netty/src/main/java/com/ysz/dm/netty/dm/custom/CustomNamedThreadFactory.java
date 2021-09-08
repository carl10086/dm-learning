package com.ysz.dm.netty.dm.custom;

import com.google.common.base.Preconditions;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 自定义前缀线程池
 */
public class CustomNamedThreadFactory implements ThreadFactory {


  private static final AtomicInteger poolNumber = new AtomicInteger(1);
  private final AtomicInteger threadNumber = new AtomicInteger(1);
  private final ThreadGroup group;
  private final String namePrefix;
  private final boolean isDaemon;

  public CustomNamedThreadFactory(final String prefix, final boolean daemon) {
    Preconditions.checkNotNull(prefix);

    /*1. 用来获取当前 线程的 thread Groups*/
    SecurityManager s = System.getSecurityManager();
    group = (s != null) ? s.getThreadGroup() : Thread.currentThread().getThreadGroup();
    /*2. prefix + 一个自增的计数*/
    namePrefix = prefix + "-" + poolNumber.getAndIncrement() + "-thread-";
    isDaemon = daemon;
  }


  @Override
  public Thread newThread(final Runnable r) {
    Thread t = new Thread(group, r, namePrefix + threadNumber.getAndIncrement(), 0);
    t.setDaemon(isDaemon);
    if (t.getPriority() != Thread.NORM_PRIORITY) {
      t.setPriority(Thread.NORM_PRIORITY);
    }
    return t;
  }
}

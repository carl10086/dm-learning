package com.ysz.dm.netty.custom.sofa.bolt.remoting.infra.thread;

import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

public class MyNamedThreadFactory implements ThreadFactory {

  private static final AtomicInteger poolNumber = new AtomicInteger(1);
  private final AtomicInteger threadNumber = new AtomicInteger(1);
  private final ThreadGroup group;
  private final String namePrefix;
  private final boolean isDaemon;

  public MyNamedThreadFactory(String prefix, boolean daemon) {
    SecurityManager s = System.getSecurityManager();
    group = (s != null) ? s.getThreadGroup() : Thread.currentThread().getThreadGroup();
    namePrefix = prefix + "-" + poolNumber.getAndIncrement() + "-thread-";
    isDaemon = daemon;
  }

  @Override
  public Thread newThread(final Runnable r) {
    Thread t = new Thread(group, r, namePrefix + threadNumber.getAndIncrement(), 0);
    t.setDaemon(isDaemon);
    if (t.getPriority() != Thread.NORM_PRIORITY) {
      /*强制优先级是 5*/
      t.setPriority(Thread.NORM_PRIORITY);
    }
    return t;
  }
}

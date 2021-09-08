package com.ysz.dm.netty.custom.netty.core.channel.eventloop.impl;

import com.ysz.dm.netty.custom.netty.core.channel.eventloop.MySingleThreadEventLoop;
import com.ysz.dm.netty.custom.netty.core.channel.eventloop.selection.MySelectedSelectionKeySet;
import io.netty.channel.ChannelException;
import io.netty.util.internal.PlatformDependent;
import io.netty.util.internal.SystemPropertyUtil;
import io.netty.util.internal.logging.InternalLogger;
import io.netty.util.internal.logging.InternalLoggerFactory;
import java.io.IOException;
import java.nio.channels.Selector;
import java.nio.channels.spi.SelectorProvider;
import java.security.AccessController;
import java.security.PrivilegedAction;

public class MyNioEventLoop extends MySingleThreadEventLoop {

  private static final InternalLogger logger = InternalLoggerFactory
      .getInstance(MyNioEventLoop.class);

  private static final int CLEANUP_INTERVAL = 256;
  private static final int MIN_PREMATURE_SELECTOR_RETURNS = 3;
  private static final int SELECTOR_AUTO_REBUILD_THRESHOLD;

  private static final boolean DISABLE_KEY_SET_OPTIMIZATION =
      SystemPropertyUtil.getBoolean("io.netty.noKeySetOptimization", false);


  /**
   * 这里的核心参数用来解决 JDK NIO 的 bug
   *
   * 可以参考:
   *  - http://bugs.sun.com/view_bug.do?bug_id=6427854
   *  - https://github.com/netty/netty/issues/203
   */
  static {
    final String key = "sun.nio.ch.bugLevel";
    final String bugLevel = SystemPropertyUtil.get(key);

    /*1. 如果 bugLevel 没有设置，强行写一个 level 进去*/
    if (bugLevel == null) {
      try {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
          System.setProperty(key, "");
          return null;
        });
      } catch (final SecurityException e) {
        logger.debug("Unable to get/set System Property: " + key, e);
      }
    }


    /*2. 读取配置, 这个配置就是 epoll 空转轮询判断的 阈值，默认是 512*/
    int selectorAutoRebuildThreshold = SystemPropertyUtil
        .getInt("io.netty.selectorAutoRebuildThreshold", 512);

    /*2.1 如果强行配置了 < 3 的， 设置 selectorAutoRebuildThreshold = 0*/
    if (selectorAutoRebuildThreshold < MIN_PREMATURE_SELECTOR_RETURNS) {
      selectorAutoRebuildThreshold = 0;
    }

    SELECTOR_AUTO_REBUILD_THRESHOLD = selectorAutoRebuildThreshold;

    /*3. 打印下 debug 信息*/
    if (logger.isDebugEnabled()) {
      logger.debug("-Dio.netty.noKeySetOptimization: {}", DISABLE_KEY_SET_OPTIMIZATION);
      logger.debug("-Dio.netty.selectorAutoRebuildThreshold: {}", SELECTOR_AUTO_REBUILD_THRESHOLD);
    }
  }


  private Selector selector;
  private Selector unwrappedSelector;


  /**
   * <pre>
   *   jdk  封装的 provider 适配各种操作系统
   * </pre>
   */
  private SelectorProvider provider;


  private static class SelectorTuple {

    final Selector unwrappedSelector;
    final Selector selector;

    SelectorTuple(Selector unwrappedSelector) {
      this.unwrappedSelector = unwrappedSelector;
      this.selector = unwrappedSelector;
    }

    SelectorTuple(Selector unwrappedSelector, Selector selector) {
      this.unwrappedSelector = unwrappedSelector;
      this.selector = selector;
    }
  }


  private SelectorTuple openSelector() {
    final Selector unwrappedSelector;
    try {
      unwrappedSelector = provider.openSelector();
    } catch (IOException e) {
      throw new ChannelException("failed to open a new selector", e);
    }

    /*1. 如果关闭了 SelectionKey 的优化，直接返回*/
    if (DISABLE_KEY_SET_OPTIMIZATION) {
      return new SelectorTuple(unwrappedSelector);
    }

    /*2. 加载 sun.nio.ch.SelectorImpl 的 class 信息*/
    Object maybeSelectorImplClass = AccessController.doPrivileged(new PrivilegedAction<Object>() {
      @Override
      public Object run() {
        try {
          return Class.forName(
              "sun.nio.ch.SelectorImpl",
              false,
              PlatformDependent.getSystemClassLoader());
        } catch (Throwable cause) {
          return cause;
        }
      }
    });

    if (!(maybeSelectorImplClass instanceof Class) ||
        // ensure the current selector implementation is what we can instrument.
        !((Class<?>) maybeSelectorImplClass).isAssignableFrom(unwrappedSelector.getClass())) {
      if (maybeSelectorImplClass instanceof Throwable) {
        /*3. 如果是异常，直接返回，优化不了*/
        Throwable t = (Throwable) maybeSelectorImplClass;
        logger.trace("failed to instrument a special java.util.Set into: {}", unwrappedSelector, t);
      }
      return new MyNioEventLoop.SelectorTuple(unwrappedSelector);
    }


    /*4. 这里可以确定一定是个 Class 了*/
    final Class<?> selectorImplClass = (Class<?>) maybeSelectorImplClass;

    /*5. 这个东西其实不是 set，一个简单的数组对象*/
    final MySelectedSelectionKeySet selectedKeySet = new MySelectedSelectionKeySet();

    return null;

  }
}

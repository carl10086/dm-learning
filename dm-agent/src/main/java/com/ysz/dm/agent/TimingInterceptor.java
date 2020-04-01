package com.ysz.dm.agent;

import java.lang.reflect.Method;
import java.util.concurrent.Callable;
import net.bytebuddy.implementation.bind.annotation.Origin;
import net.bytebuddy.implementation.bind.annotation.RuntimeType;
import net.bytebuddy.implementation.bind.annotation.SuperCall;

public class TimingInterceptor {

  @RuntimeType
  public static Object intercept(@Origin Method method,
      // 调用该注解后的Runnable/Callable，会导致调用被代理的非抽象父方法
      @SuperCall Callable<?> callable) throws Exception {
    long start = System.currentTimeMillis();
    try {
      return callable.call();
    } finally {
      System.out.println(method + " took " + (System.currentTimeMillis() - start));
    }
  }

}

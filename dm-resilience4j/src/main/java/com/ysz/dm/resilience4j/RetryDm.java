package com.ysz.dm.resilience4j;

import com.google.common.collect.Lists;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Callable;

public class RetryDm {

  public static void main(String[] args) throws Exception {
    RetryConfig config = RetryConfig.custom()
        .maxAttempts(2)
        .waitDuration(Duration.ofMillis(100))
//        .retryOnResult(response -> response.getStatus() == 500)
//        .retryOnException(e -> e instanceof WebServiceException)
//        .retryExceptions(IOException.class, TimeoutException.class)
//        .ignoreExceptions(BusinessException.class, OtherBusinessException.class)
//        .failAfterMaxAttempts(true)
        .build();

    RetryRegistry registry = RetryRegistry.of(config);
    Retry tst = registry.retry("tst");
    final List<Long> params = Lists.newArrayList(1L, 2L);
//    String s = tst.executeSupplier(() -> mockBiz(params));
    String s = tst.executeCallable(new Callable<String>() {
      @Override
      public String call() throws Exception {
        return mockBiz(params);
      }
    });
    System.err.println(s);
  }

  public static String mockBiz(List<Long> ids) {
    System.err.println("执行一次 mockBiz");
//    sleep(200);
    if (ids.size() == 2) {
      throw new RuntimeException("size is 2");
    }
    return ids + "";
  }


  private static void sleep(long mills) {
    try {
      Thread.sleep(mills);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }
}

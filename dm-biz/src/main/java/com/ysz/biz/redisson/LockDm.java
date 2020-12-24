package com.ysz.biz.redisson;

import java.util.concurrent.TimeUnit;
import org.redisson.Redisson;
import org.redisson.api.RReadWriteLock;
import org.redisson.api.RedissonClient;
import org.redisson.config.Config;

public class LockDm {

  public static void main(String[] args) throws Exception {
    Config config = new Config();
    config.useSingleServer().setAddress("redis://127.0.0.1:6379");

    RedissonClient redisson = Redisson.create(config);

    final RReadWriteLock myLock = redisson.getReadWriteLock("myLock");
    try {
      final boolean b = myLock.readLock().tryLock(1L, 2L, TimeUnit.SECONDS);
      Thread.sleep(5L);
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
    }
    myLock.readLock().unlock();

    redisson.shutdown();
  }
}

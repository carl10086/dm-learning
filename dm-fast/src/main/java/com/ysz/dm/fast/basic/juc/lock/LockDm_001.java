package com.ysz.dm.fast.basic.juc.lock;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class LockDm_001 {


  private Lock lock = new ReentrantLock(false);

  public void execute() {
    if (lock.tryLock()) {
      System.out.println("获取锁成功");
      try {

        try {
          Thread.sleep(2000L);
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
      } finally {
        System.out.println("释放锁成功");
        lock.unlock();
      }
    } else {
      System.out.println("获取锁失败");
    }
  }

  public static void main(String[] args) throws Exception {
    final LockDm_001 lockDm_001 = new LockDm_001();
    final Thread t1 = new Thread(lockDm_001::execute);
    final Thread t2 = new Thread(lockDm_001::execute);

    t1.start();
    t2.start();
    t1.join();
    t2.join();

    lockDm_001.execute();


  }


}

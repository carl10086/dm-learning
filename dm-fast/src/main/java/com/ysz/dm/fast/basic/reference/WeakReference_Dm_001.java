package com.ysz.dm.fast.basic.reference;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;

/**
 * @author carl
 */
public class WeakReference_Dm_001 {

  public static int M = 1024 * 1024;

  public static void printlnMemory(String tag) {
    Runtime runtime = Runtime.getRuntime();
    int M = WeakReference_Dm_001.M;
    System.out.println("\n" + tag + ":");
    System.out
        .println(runtime.freeMemory() / M + "M(free)/" + runtime.totalMemory() / M + "M(total)");
  }

  public static void main(String[] args) {
    WeakReference_Dm_001.printlnMemory("1.原可用内存和总内存");

    ReferenceQueue<Object> referenceQueue = new ReferenceQueue<>();
    WeakReference<Object> weakRerference;
    //创建弱引用
    byte[] bytes = new byte[10 * WeakReference_Dm_001.M];
    /*测试下是否有 referenceQueue 的区别*/
//    weakRerference = new WeakReference<>(bytes, referenceQueue);
    weakRerference = new WeakReference<>(bytes);
    /*..引发 gc.*/
    bytes = null;
    WeakReference_Dm_001.printlnMemory("2.实例化10M的数组,并建立弱引用");
    System.out.println("weakRerference.get() : " + weakRerference.get());

    System.gc();
    WeakReference_Dm_001.printlnMemory("3.GC后");
    System.out.println("weakRerference.get() : " + weakRerference.get());
  }

}

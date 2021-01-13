package com.ysz.dm.fast.jvm;

/**
 * 模拟 由于 Xms != Xmx 导致启动的时候大量 gc
 *
 * 可以通过 jstat -gc 以及观察日志来获取
 *
 * -Xms10m -Xmx200m -XX:NewRatio=1  -XX:+UseConcMarkSweepGC -XX:CMSInitiatingOccupancyFraction=75 -XX:+UseCMSInitiatingOccupancyOnly -XX:+ExplicitGCInvokesConcurrent -XX:+ParallelRefProcEnabled -XX:+CMSParallelInitialMarkEnabled -XX:MaxTenuringThreshold=3 -XX:+UnlockDiagnosticVMOptions -XX:ParGCCardsPerStrideChunk=1024 -Xloggc:/Users/carl/tmp/useless/gc.log -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintPromotionFailure -XX:+PrintGCApplicationStoppedTime
 */
public class JvmGcDm_001_DynamicExpand {


  public static void main(String[] args) throws Exception {
    /*等待开启 jstat 命令*/
    System.out.println("start");
    System.in.read();
    for (int i = 0; i < 10000; i++) {
      AllocateHelper.allo(i % 200 == 0);
    }
    System.out.println("end");
  }

}

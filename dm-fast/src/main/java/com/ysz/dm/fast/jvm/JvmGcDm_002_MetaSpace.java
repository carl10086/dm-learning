package com.ysz.dm.fast.jvm;

import groovy.lang.GroovyClassLoader;
import java.lang.reflect.Method;

/**
 * 使用 ClassLoader 加载执行 方法、造成 Metaspace 溢出 .
 *
 * -Xms200m -Xmx200m  -XX:MetaspaceSize=32m -XX:MaxMetaspaceSize=32m -XX:NewRatio=1  -XX:+UseConcMarkSweepGC -XX:CMSInitiatingOccupancyFraction=75 -XX:+UseCMSInitiatingOccupancyOnly -XX:+ExplicitGCInvokesConcurrent -XX:+ParallelRefProcEnabled -XX:+CMSParallelInitialMarkEnabled -XX:MaxTenuringThreshold=3 -XX:+UnlockDiagnosticVMOptions -XX:ParGCCardsPerStrideChunk=1024 -Xloggc:/Users/carl/tmp/useless/gc.log -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintPromotionFailure -XX:+PrintGCApplicationStoppedTime
 *
 */
public class JvmGcDm_002_MetaSpace {

  private static String script(int i) {
    return "class GroovySimpleFileCreator" + i + " {\n"
        + "    public int incr(int i) {\n"
        + "        return i + 1;\n"
        + "    }\n"
        + "}";
  }

  public static void main(String[] args) throws Exception {
    GroovyClassLoader loader = new GroovyClassLoader();
    for (int i = 0; i < 5_000; i++) {
      final Class aClass = loader.parseClass(script(i));
      if (i % 50 == 0) {
        System.err.println(i);
        // loader = new GroovyClassLoader(); /*注释这一行、会发现爆出 metaSpace OOM, 说明 类元信息的生命周期和 ClassLoader 一致*/
      }
      final Method incr = aClass.getDeclaredMethod("incr", int.class);
      if (i == 1900) {
        System.out.println("input then go on");
        System.in.read();
      }
      incr.setAccessible(true);
      incr.invoke(aClass.newInstance(), 3);
    }

  }
}

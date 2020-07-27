package com.ysz.dm.fast.basic.gc;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import static java.util.UUID.randomUUID;

public class SystemGc_Dm_001 {

  private static final Map<String, String> cache = new HashMap<>();

  public static void main(String[] args) {
    Scanner scanner = new Scanner(System.in);

    while (scanner.hasNext()) {
      final String next = scanner.next();
      if ("fill".equals(next)) {
        // 放入 100W 个字符串
        for (int i = 0; i < 1_000_000; i++) {
          cache.put(randomUUID().toString(), randomUUID().toString());
        }
      } else if ("invalidate".equals(next)) {
        // 清空缓存
        cache.clear();
      } else if ("gc".equals(next)) {
        // 显示 gc
        System.gc();
      } else if ("exit".equals(next)) {
        // 退出程序
        System.exit(0);
      } else {
        System.out.println("unknown");
      }
    }
  }
}

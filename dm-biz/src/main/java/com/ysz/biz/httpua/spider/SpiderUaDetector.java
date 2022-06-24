package com.ysz.biz.httpua.spider;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

public class SpiderUaDetector {


  public static void main(String[] args) throws Exception {
    Map<SpiderEnum, Integer> map = new HashMap<>();
    for (SpiderEnum value : SpiderEnum.values()) {
      map.put(value, 0);
    }
    final LineIterator lineIterator = FileUtils.lineIterator(new File("/Users/carl/work/dt/tmp/ua.txt"));
    while (lineIterator.hasNext()) {
      final String line = lineIterator.nextLine();
      final Set<SpiderEnum> spiderEnums = SpiderEnum.detectAll(line);
      if (spiderEnums.size() == 0) {
        continue;
      }
      if (spiderEnums.size() == 1) {
        final SpiderEnum spiderEnum = spiderEnums.stream().findFirst().get();
//        System.out.printf("%s:%s\n", Lists.newArrayList(spiderEnums).get(0), line);
        map.put(spiderEnum, map.get(spiderEnum) + 1);
      } else {
        System.err.println(spiderEnums);
        System.err.println(line);
      }


    }
    map.forEach((spiderEnum, integer) -> System.out.printf("%s:\t%s\n", spiderEnum, integer));
    final int sum = map.entrySet().stream().map(Entry::getValue).mapToInt(x -> x).sum();
    System.out.println(sum);
  }
}

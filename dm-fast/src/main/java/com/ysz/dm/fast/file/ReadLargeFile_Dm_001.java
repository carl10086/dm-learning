package com.ysz.dm.fast.file;

import it.unimi.dsi.fastutil.objects.ObjectOpenHashBigSet;
import java.io.File;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

public class ReadLargeFile_Dm_001 {

  public static void main(String[] args) throws Exception {
//    IntOpenHashBigSet set = new IntOpenHashBigSet();
    ObjectOpenHashBigSet<String> set = new ObjectOpenHashBigSet<>();
    long start = System.currentTimeMillis();
    String theFile = "/Users/carl/tmp/1.res";
    LineIterator it = FileUtils.lineIterator(new File(theFile), "UTF-8");
    try {
      while (it.hasNext()) {
        String line = it.nextLine();
        if (line != null && !"".equalsIgnoreCase(line)) {
          set.add(line);
        }
      }
    } finally {
      LineIterator.closeQuietly(it);
    }
    ObjectOpenHashBigSet<String> set2 = new ObjectOpenHashBigSet<>(set);
    int cnt = 0;
    for (String data : set2) {
      if (set.contains(data)) {
        cnt++;
      }
    }
    System.out.println(set2.size64());
    System.out.println(System.currentTimeMillis() - start);


  }

}

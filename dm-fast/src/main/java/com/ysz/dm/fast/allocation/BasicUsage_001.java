package com.ysz.dm.fast.allocation;

import com.google.monitoring.runtime.instrumentation.AllocationRecorder;
import java.util.ArrayList;
import java.util.List;

/**
 * @author carl
 */
public class BasicUsage_001 {

  private List<Byte[]> cache = new ArrayList<>();

  private Byte[] kilos() {
    Byte[] bytes = new Byte[1024];
    for (int i = 0; i < bytes.length; i++) {
      bytes[i] = Byte.valueOf((byte) 1);
    }
    return bytes;
  }

  public static void main(String[] args) throws Exception {
    new BasicUsage_001().test();
  }

  private void test() throws Exception {
    AllocationRecorder.addSampler((count, desc, newObj, size) -> {
      System.out.println("I just allocated the object " + newObj
          + " of type " + desc + " whose size is " + size);
      if (count != -1) {
        System.out.println("It's an array of size " + count);
      }
    });
    for (int i = 0; i < 10; i++) {
      new String("foo");
    }
  }

}

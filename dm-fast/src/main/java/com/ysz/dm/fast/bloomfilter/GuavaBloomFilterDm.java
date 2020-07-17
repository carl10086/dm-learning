package com.ysz.dm.fast.bloomfilter;

import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnels;

/**
 * @author carl
 */
public class GuavaBloomFilterDm {


  public void simple() {
    BloomFilter<Integer> objectBloomFilter = BloomFilter.create(Funnels.integerFunnel(), 10, 0.01);
    objectBloomFilter.put(10);
    System.err.println(objectBloomFilter.put(10));
  }

  public static void main(String[] args) {
    new GuavaBloomFilterDm().simple();
  }
}

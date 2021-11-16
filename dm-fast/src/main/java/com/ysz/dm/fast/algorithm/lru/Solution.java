package com.ysz.dm.fast.algorithm.lru;


import java.util.ArrayList;
import java.util.Comparator;

/**
 * https://www.nowcoder.com/practice/e3769a5f49894d49b871c09cadd13a61?tpId=117&&tqId=37804&rp=1&ru=/activity/oj&qru=/ta/job-code-high/question-ranking
 */
public class Solution {

  public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
    if (k == 0) {
      return new ArrayList<>();
    }

    java.util.PriorityQueue<Integer> queue = new java.util.PriorityQueue<>(k, Comparator.reverseOrder());
    for (int i : input) {
      queue.add(i);
      if (queue.size() == k + 1) {
        queue.poll();
      }
    }
    return new ArrayList<>(queue);
  }

  private int[] toArray(java.util.HashSet<Integer> res) {
    int[] array = new int[res.size()];
    int i = 0;
    for (Integer re : res) {
      array[i++] = re;
    }

    return array;
  }
}

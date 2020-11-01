package com.ysz.dm.fast.algorithm.heap;

import java.util.Arrays;

/**
 * https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/
 */
public class LeetCode_40 {


  public int[] getLeastNumbers(int[] arr, int k) {
    if (k <= 0) {
      return new int[0];
    }
    /*最大堆*/
    java.util.PriorityQueue<Integer> priorityQueue = new java.util.PriorityQueue<>(
        java.util.Comparator.reverseOrder());
    for (int i = 0; i < arr.length; i++) {
      int item = arr[i];
      if (priorityQueue.size() == k) {
        /*满了的时候*/
        /*这里有一个优化, 先比较, 只有 item 小的时候才有资格加入*/
        if (priorityQueue.peek() > item) {
          priorityQueue.poll();
          priorityQueue.offer(item);
        }
      } else {
        priorityQueue.offer(item);
      }
    }

    int[] res = new int[k];
    for (int i = 0; i < k; i++) {
      res[i] = priorityQueue.poll();
    }
    return res;
  }


  public static void main(String[] args) {
    System.err.println(Arrays.toString(new LeetCode_40().getLeastNumbers(
        new int[]{0, 0, 0, 2, 0, 5},
        2
    )));
  }

}

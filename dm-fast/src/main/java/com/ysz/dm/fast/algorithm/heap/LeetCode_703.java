package com.ysz.dm.fast.algorithm.heap;


/**
 * https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/
 */
public class LeetCode_703 {

  static class KthLargest {

    /*最大话、 默认的最小堆就可以了*/
    private java.util.PriorityQueue<Integer> queue = new java.util.PriorityQueue<>();
    private int k;

    public KthLargest(int k, int[] nums) {
      this.k = k;
      for (int num : nums) {
        addToQueue(num);
      }
    }


    private void addToQueue(int i) {
      final int size = this.queue.size();
      if (size == k) {
        if (queue.peek() < i) {
          queue.poll();
          queue.offer(i);
        }
      } else {
        queue.offer(i);
      }

    }

    public int add(int val) {
      addToQueue(val);
      return this.queue.peek();
    }
  }

  public static void main(String[] args) {
    KthLargest kthLargest = new KthLargest(3, new int[]{4, 5, 8, 2});
    System.err.println(kthLargest.add(3));
    System.err.println(kthLargest.add(5));
    System.err.println(kthLargest.add(10));
    System.err.println(kthLargest.add(9));
    System.err.println(kthLargest.add(4));
  }
}

package com.ysz.dm.fast.algorithm.stack.leetcode;

public class LeetCode_232 {

  public static void main(String[] args) {
    MyQueue myQueue = new MyQueue();
    myQueue.push(1);
    myQueue.push(2);
    myQueue.push(3);
    myQueue.push(4);
    System.err.println(myQueue.pop() == 1);
    System.err.println(myQueue.empty() == false);
    System.err.println(myQueue.peek() == 2);
    System.err.println(myQueue.pop() == 2);
    System.err.println(myQueue.pop() == 3);
    System.err.println(myQueue.pop() == 4);
  }

  static class MyQueue {

    private java.util.LinkedList<Integer> first;
    private java.util.LinkedList<Integer> second;
    /*用来减少拷贝次数*/
    private boolean nowInFirst = true;

    /** Initialize your data structure here. */
    public MyQueue() {
      first = new java.util.LinkedList<>();
      second = new java.util.LinkedList<>();
    }

    /** Push element x to the back of queue. */
    public void push(int x) {
      if (!nowInFirst) {
        copy();
      }
      first.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
      if (nowInFirst) {
        copy();
      }
      return second.pop();
    }


    private void copy() {
      if (nowInFirst) {
        int size = first.size();
        for (int i = 0; i < size; i++) {
          second.push(first.pop());
        }
        nowInFirst = false;
      } else {
        int size = second.size();
        for (int i = 0; i < size; i++) {
          first.push(second.pop());
        }
        nowInFirst = true;
      }
    }

    /** Get the front element. */
    public int peek() {
      if (nowInFirst) {
        copy();
      }
      return second.peek();
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
      if (nowInFirst) {
        return first.isEmpty();
      }
      return second.isEmpty();
    }
  }
}

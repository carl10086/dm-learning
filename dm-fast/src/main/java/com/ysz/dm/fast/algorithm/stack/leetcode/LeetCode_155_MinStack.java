package com.ysz.dm.fast.algorithm.stack.leetcode;


public class LeetCode_155_MinStack {

  public static void main(String[] args) {
    MinStack stack = new MinStack();
    stack.push(-2);
    stack.push(0);
    stack.push(-3);
    System.err.println(stack.getMin());
    stack.pop();
    stack.pop();
    System.err.println(stack.getMin());
  }

  static class MinStack {

    /*每次增加都记录最小值、大家一起走*/
    private java.util.LinkedList<Integer> minStack;
    private java.util.LinkedList<Integer> stack;

    /** initialize your data structure here. */
    public MinStack() {
      this.minStack = new java.util.LinkedList<>();
      stack = new java.util.LinkedList<>();
      this.minStack.push(Integer.MAX_VALUE);
    }

    public void push(int x) {
      this.stack.push(x);
      this.minStack.push(Math.min(x, minStack.peek()));
    }

    public void pop() {
      this.stack.pop();
      this.minStack.pop();
    }

    public int top() {
      return this.stack.peek();
    }

    public int getMin() {
      return minStack.peek();
    }
  }
}

package com.ysz.dm.fast.basic.juc.cas;

import java.util.concurrent.atomic.AtomicReference;

public class ConcurrentStack<E> {

  private AtomicReference<Node<E>> top = new AtomicReference<>();

  private static class Node<E> {

    public E item;
    public Node<E> next;

    public Node() {
    }

    public Node(final E item) {
      this.item = item;
    }
  }

  /**
   * @desc: stack 并发争夺在于 top
   *  1.push 只要替换 top
   */
  public void push(E item) {
    Node<E> newHead = new Node<E>(item);
    Node<E> oldHead;

    do {
      oldHead = top.get();
      newHead.next = oldHead;
    } while (!top.compareAndSet(oldHead, newHead));
  }

  public E pop() {
    Node<E> oldHead;
    Node<E> newHead;
    do {
      oldHead = top.get();
      if (oldHead == null) {
        return null;
      }
      newHead = oldHead.next;
    } while (!top.compareAndSet(oldHead, newHead));

    return oldHead.item;
  }
}

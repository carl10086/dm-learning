package com.ysz.dm.fast.algorithm.lru;

import java.util.HashMap;
import java.util.Map;

public class LRUCache {


  private static final int NOT_EXIST = -1;

  private Node first;
  private Node last;


  private int size = 0;
  private int capacity;

  /*双向链表的 hash 索引*/
  private Map<Integer, Node> hashIndex;

  public LRUCache(int capacity) {
    this.hashIndex = new HashMap<>(capacity);
    this.capacity = capacity;
    /*双向链表的 tail 保留最新的数据*/
  }

  public int get(int key) {
    if (size == 0) {
      return NOT_EXIST;
    }
    Node node = hashIndex.get(key);
    if (node == null) {
      return NOT_EXIST;
    } else {
      /*刚访问的数据要移动 tail*/
      moveNodeToLast(node);
      return node.data;
    }
  }

  /**
   * 调用当前方法必须保证 size > 0 first , last != null
   * @param node 要移动的合法结点
   */
  private void moveNodeToLast(Node node) {
    if (node == last) {
      /*1. 边界1: 是 last*/
      return;
    } else if (first == node) {
      /*2. 边界: 是 first; */
      /*此时 first != last, 否则 上面 node == last 条件必须会满足, 也就是说 firstNext 不可能是 null*/
      Node next = node.next;
      next.prev = null;
      this.first = next;
      node.prev = this.last;
      this.last.next = node;
      node.next = null;
      this.last = node;
    } else {
      /*此时 node 必须在中间 , first, ...node..., last*/
      Node prev = node.prev;
      Node next = node.next;
      prev.next = next;
      next.prev = prev;

      node.prev = this.last;
      this.last.next = node;
      node.next = null;
      this.last = node;
    }

  }

  public void put(int key, int value) {
    /*1. 边界1 空*/
    if (size == 0) {
      addWhenEmpty(key, value);
    } else {
      Node node = hashIndex.get(key);
      if (node != null) {
        /*查找到了结点直接返回*/
        moveNodeToLast(node);
        node.data = value;
      } else {
        if (size < capacity) {
          addNewNodeToLast(key, value);
        } else {
          /*这里是已经满了情况, 要删除 first*/
          int toDelKey = this.first.key;
          this.hashIndex.remove(toDelKey);
          this.size--;
          if (size == 0) {
            /*边界1, capacity = size = 1, 还要删除以前的节点, 因为size-- 了，所以这里的判断是0*/
            addWhenEmpty(key, value);
            return;
          } else {
            this.first = this.first.next;
            this.first.prev = null;
            addNewNodeToLast(key, value);
          }
        }
      }
    }
  }

  private void addNewNodeToLast(int key, int value) {
    Node node;
    node = new Node(value, key, null, null);
    this.hashIndex.put(key, node);
    appendNowNodeToLast(node);
    this.size++;
  }

  private void addWhenEmpty(int key, int value) {
    Node node = new Node(value, key, null, null);
    this.first = node;
    this.last = node;
    this.hashIndex.put(key, node);
    this.size = 1;
  }

  private void appendNowNodeToLast(Node node) {
    this.last.next = node;
    node.prev = this.last;
    this.last = node;
  }


  /**
   * 双向链表的内部结点
   * 也是 HashMap 的val
   */
  static class Node {

    int data;
    Node prev;
    Node next;


    int key;

    public Node(int data, int key, Node prev, Node next) {
      this.data = data;
      this.key = key;
      this.prev = prev;
      this.next = next;
    }
  }

  public static void main(String[] args) {

    LRUCache cache = new LRUCache(2 /* 缓存容量 */);

    cache.put(1, 1);
    cache.put(2, 2);
    cache.get(1);       // 返回  1
    cache.put(3, 3);    // 该操作会使得关键字 2 作废
    cache.get(2);       // 返回 -1 (未找到)
    cache.put(4, 4);    // 该操作会使得关键字 1 作废
    cache.get(1);       // 返回 -1 (未找到)
    cache.get(3);       // 返回  3
    cache.get(4);       // 返回  4
  }
}

package com.ysz.biz.hanlp.learn.dict.bin;

import java.util.AbstractMap;
import org.jetbrains.annotations.NotNull;

/**
 *
 */
public abstract class BaseNode<V> implements Comparable<BaseNode<V>> {

  /**
   * 子节点,
   */
  protected BaseNode[] children;

  /**
   * 节点状态
   */
  protected Status status;

  /**
   * 节点代表的字符
   */
  protected char c;

  /**
   * 节点代表的值
   */
  protected V value;

  public abstract BaseNode<V> child(char c);

  /**
   * 增加一个子节点
   *
   * @param node node
   */
  public abstract boolean addChild(BaseNode<V> node);

  public boolean hasChild(char c) {
    return child(c) != null;
  }


  public BaseNode<V> transition(String path, int begin) {
    return transition(path.toCharArray(), begin);
  }

  public BaseNode<V> transition(char[] path, int begin) {

    BaseNode<V> cur = this;
    for (int i = begin; i < path.length; ++i) {
      /*寻找到 char 的对应 Node*/
      cur = cur.child(path[i]);
      if (notExist(cur)) {
        return null;
      }
    }
    return cur;
  }

  public BaseNode<V> transition(char path) {
    BaseNode<V> cur = this.child(path);
    return notExist(cur) ? null : cur;
  }

  private static boolean notExist(BaseNode node) {
    return node == null || node.status == Status.UNDEFINED_0;
  }

  private static boolean exist(BaseNode node) {
    return !notExist(node);
  }

  public enum Status {
    /**
     * 未指定，用于删除词条
     */
    UNDEFINED_0,
    /**
     * 不是词语的结尾
     */
    NOT_WORD_1,
    /**
     * 是个词语的结尾，并且还可以继续
     */
    WORD_MIDDLE_2,
    /**
     * 是个词语的结尾，并且没有继续
     */
    WORD_END_3,
  }


  public class TrieEntry extends AbstractMap.SimpleEntry<String, V> implements Comparable<TrieEntry> {

    public TrieEntry(String key, V value) {
      super(key, value);
    }

    @Override
    public int compareTo(TrieEntry o) {
      return getKey().compareTo(o.getKey());
    }
  }


  public Status getStatus() {
    return status;
  }

  public char getC() {
    return c;
  }

  public V getValue() {
    return value;
  }

  public void setStatus(Status status) {
    this.status = status;
  }

  public void setC(char c) {
    this.c = c;
  }

  public void setValue(V value) {
    this.value = value;
  }

  @Override
  public int compareTo(@NotNull BaseNode o) {
    return compareToChar(o.c);
  }

  public int compareToChar(char other) {
    if (this.c > other) {
      return 1;
    }
    if (this.c < other) {
      return -1;
    }
    return 0;
  }
}


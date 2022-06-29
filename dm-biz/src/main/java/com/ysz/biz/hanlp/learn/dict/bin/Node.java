package com.ysz.biz.hanlp.learn.dict.bin;


/**
 * 深度 > 2 的 node, 由于第一层的 node 走一个全量的内存 char 数组
 */
public class Node<V> extends BaseNode<V> {


  public Node() {
  }

  public Node(char c, Status status, V value) {
    this.c = c;
    this.status = status;
    this.value = value;
  }

  @Override
  public BaseNode<V> child(char c) {
    if (children == null) {
      return null;
    }
    /*二分搜索 children 中, c 的位置*/
    return ArrayTool.binarySearch(children, c) < 0 ? null : children[ArrayTool.binarySearch(children, c)];
  }

  /**
   * 这个状态机特别复杂
   *
   * @param node node
   */
  @Override
  public boolean addChild(BaseNode<V> node) {
    boolean add = false;

    if (children == null) {
      children = new BaseNode[0];
    }

    int index = ArrayTool.binarySearch(children, node);
    if (index >= 0) {
      /*found */

      final BaseNode target = children[index];

      /*判断的是我们想插入节点的状态. 需要和 target 节点状态进行对比, 再决定*/
      switch (node.status) {
        case UNDEFINED_0:
          /*node 这个状态, 表名之前删除过.  target 的状态如果不是结尾*/
          if (target.status != Status.NOT_WORD_1) {
            target.status = Status.NOT_WORD_1;
            target.value = null;
            add = true;
          }
          break;
        case NOT_WORD_1:
          if (target.status == Status.WORD_END_3) {
            target.status = Status.WORD_MIDDLE_2;
          }
          break;
        case WORD_MIDDLE_2:
          break;
        case WORD_END_3:
          if (target.status != Status.WORD_END_3) {
            target.status = Status.WORD_MIDDLE_2;
          }
          if (target.getValue() == null) {
            add = true;
          }
          target.setValue(node.getValue());
          break;
      }


    }

    return false;
  }
}

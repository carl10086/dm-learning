package com.ysz.dm.fast.algorithm.linked;

import com.google.common.base.Joiner;
import java.util.ArrayList;
import java.util.List;

public class ListNode {

  int val;
  ListNode next;

  public ListNode(int x) {
    val = x;
  }

  ListNode() {
  }


  public ListNode(int val, ListNode next) {
    this.val = val;
    this.next = next;
  }


  public void show() {
    List<Integer> resultList = new ArrayList<>();

    ListNode cur = this;
    while (cur != null) {
      resultList.add(cur.val);
      cur = cur.next;
    }

    System.out.println(Joiner.on("->").join(resultList));

  }
}

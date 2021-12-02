package com.ysz.dm.fast.algorithm.linked;

import com.google.common.collect.Lists;
import java.util.List;

public class ReversedList_001 {


  public ListNode ReverseList(ListNode head) {
    if (head == null) {
      return null;
    }

    ListNode prev = head;
    ListNode current = prev.next;

    /*1. 后面没了*/
    if (current == null) {
      return prev;
    }

    /*2. 有环?.. 可以不考虑*/
    if (prev == current) {
      return head;
    }

    prev.next = null;

    /*3. 递归的 反转 next*/
    final ListNode reversedCurrent = ReverseList(current);
    current.next = prev;
    return reversedCurrent;
  }


  private static ListNode mock(List<Integer> list) {
    ListNode root = new ListNode(list.get(0));
    ListNode cur = root;

    for (int i = 1; i < list.size(); i++) {
      cur.next = new ListNode(list.get(i));
      cur = cur.next;
    }

    return root;
  }


  public static void main(String[] args) {

    final ListNode mock = mock(Lists.newArrayList(7, 3, 4, 1, 2, 5));

    final ListNode execute = new ReversedList_001().ReverseList(mock);

    execute.show();
  }


}

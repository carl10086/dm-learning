package com.ysz.dm.fast.algorithm.linked;

import java.util.List;

public class Solution {


  public static void main(String[] args) {
    ListNode l1 = newListNode(1, 2, 3, 4, 5, 6);

    ListNode listNode = new Solution().middleNode(l1);
    System.err.println(1);
  }

  public ListNode removeNthFromEnd(ListNode head, int n) {
    /*因为题目中可以保证 n 有效*/
    ListNode slowPrev = null; /*最后删除需要知道前一个节点是什么*/
    ListNode slow = head;
    ListNode fast = head;
    for (int i = 0; i < n; i++) {
      if (fast.next == null) {
        return head.next;
      } else {
        fast = fast.next;
      }
    }

    while (true) {
      if (fast == null) {
        break;
      }
      slowPrev = slow;
      slow = slow.next;
      fast = fast.next;
    }
    slowPrev.next = slow.next;
    return head;
  }


  public ListNode middleNode(ListNode head) {
    if (head == null) {
      return null;
    }

    if (head.next == null) {
      return head;
    }

    if (head.next.next == null) {
      return head.next;
    }


    /*至少3个元素*/

    /*一次走一下*/
    ListNode slow = head.next;
    /*一次走2下*/
    ListNode fast = head.next.next;

    while (true) {
      if (fast.next == null) {
        return slow;
      }
      if (fast.next.next == null) {
        return slow.next;
      }
      slow = slow.next;
      fast = fast.next.next;
    }
  }

  public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null) {
      return l2;
    }

    if (l2 == null) {
      return l1;
    }

    ListNode first = l1;
    ListNode second = l2;

    /*作为哨兵、后面不要即可*/
    ListNode resNode = new ListNode();
    ListNode currResNode = resNode;
    while (true) {
      int l1Val = first.val;
      int l2Val = second.val;
      if (l1Val < l2Val) {
        // 取 first 当前节点的下一个节点、先保存先
        ListNode firstNext = first.next;
        first.next = null;
        currResNode.next = first;
        currResNode = first;
        first = firstNext;
      } else {
        ListNode secNext = second.next;
        second.next = null;
        currResNode.next = second;
        currResNode = second;
        second = secNext;
      }

      /*然后判断边界条件*/

      /*1. 边界1、二者都为空*/
      if (first == null && second == null) {
        break;
      } else if (
          first == null && second != null
      ) {
        currResNode.next = second;
        break;
      } else if (first != null && second == null) {
        currResNode.next = first;
        break;
      }
    }
    /*去除哨兵*/
    return resNode.next;
  }

  public static ListNode newListNode(int... items) {
    final ListNode first = new ListNode(items[0]);
    ListNode cur = first;
    int length = items.length;
    for (int i = 1; i < length; i++) {
      ListNode newNode = new ListNode(items[i]);
      cur.next = newNode;
      cur = newNode;
    }
    return first;
  }


}

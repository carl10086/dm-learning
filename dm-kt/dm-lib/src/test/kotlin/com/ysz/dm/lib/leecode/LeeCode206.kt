package com.ysz.dm.lib.leecode

/**
 * @author carl
 * @since 2023-02-09 6:28 PM
 **/
class LeeCode206 {
    class ListNode(var `val`: Int) {
        var next: ListNode? = null
    }

    fun reverseList(head: ListNode?): ListNode? {
        if (head == null) return null
        else if (head.next == null) return head
        else {
            var slow = head
            var fast = head.next!!
            return fast
        }
    }
}


fun main(args: Array<String>) {

}
package com.ysz.dm.lib.leecode

/**
 * 对应题目为 https://leetcode.cn/problems/maximum-number-of-consecutive-values-you-can-make/
 * @author carl
 * @since 2023-02-04 10:59 PM
 **/
object LeeCode1798 {
    fun getMaximumConsecutive(coins: IntArray): Int {
        val sorted = coins.sorted()
        var count = 0;

        for (coin in sorted) {
            val max = count + coin

            if (count + 1 in coin..max) {
                count = max
            } else {
                break
            }
        }
        return count + 1
    }

}


fun main(args: Array<String>) {
    println(LeeCode1798.getMaximumConsecutive(intArrayOf(1, 1, 1, 4)))
}
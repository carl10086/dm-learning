package com.ysz.dm.lib.leecode

/**
 * keypoint: 让留下的最后1个值最小 .
 * @author carl
 * @since 2023-02-11 11:24 PM
 **/
class LeeCode2335 {
    private fun maxIndex(amount: IntArray, excludeIndex: Int = -1): Int {
        var max = -1
        var maxPos = -1
        amount.forEachIndexed { index, i ->
            if (excludeIndex != index && i > max) {
                max = i
                maxPos = index
            }
        }

        return maxPos
    }


    fun fillCups(amount: IntArray): Int {
        var steps = 0;
        while (true) {
            var maxIdx = maxIndex(amount)
            if (amount[maxIdx] == 0) break

            amount[maxIdx] = amount[maxIdx] - 1
            steps++

            maxIdx = maxIndex(amount, maxIdx)
            if (amount[maxIdx] == 0) {
                amount[maxIdx] = amount[maxIdx] - 1
            }
        }
        return steps
    }
}

fun main(args: Array<String>) {
    print(LeeCode2335().fillCups(intArrayOf(5, 4, 4)))
}
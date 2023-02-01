package com.ysz.dm.lib.leecode

/**
 * @author carl
 * @since 2023-01-28 12:47 AM
 **/
class LeeCode1664 {
    companion object {
        fun waysToMakeFair(nums: IntArray): Int {
            var evenSum = 0
            var oddSum = 0
            nums.forEachIndexed { index, i ->
                if (index % 2 == 0) {
                    evenSum += i
                } else {
                    oddSum += i
                }
            }
            println("origin: ${nums.contentToString()}")
            println("evenSum:$evenSum, oddSum:$oddSum")


            var res = 0

            var oddNew: Int
            var evenNew: Int

            var oddBefore = 0
            var evenBefore = 0

            var oddAfter: Int
            var evenAfter: Int


            nums.forEachIndexed { index, i ->
                oddAfter = oddSum - oddBefore
                evenAfter = evenSum - evenBefore
                if (index % 2 == 0) {
                    // 偶数
                    evenNew = evenBefore + oddAfter
                    oddNew = oddBefore + evenAfter - i

                    println("index:${index}, i:${i}, evenBefore: $evenBefore, evenAfter: $evenAfter , oddBefore:${oddBefore} , oddAfter: $oddAfter")

                    evenBefore += i
                } else {
                    // 奇数
                    evenNew = evenBefore + oddAfter - i
                    oddNew = oddBefore + evenAfter

                    println("index:${index}, i:${i}, evenBefore: $evenBefore, evenAfter: $evenAfter , oddBefore:${oddBefore} , oddAfter: $oddAfter")
                    oddBefore += i
                }



                if (oddNew == evenNew) res++

            }


            return res
        }
    }
}

fun main() {
    LeeCode1664.waysToMakeFair(intArrayOf(2, 1, 6, 4))
}
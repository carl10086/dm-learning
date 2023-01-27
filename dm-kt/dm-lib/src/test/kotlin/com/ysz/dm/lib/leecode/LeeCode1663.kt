package com.ysz.dm.lib.leecode

import com.ysz.dm.lib.leecode.LeeCode1663.Companion.codeAsChar
import com.ysz.dm.lib.leecode.LeeCode1663.Companion.getSmallestString

/**
 * https://leetcode.com/problems/smallest-string-with-a-given-numeric-value/
 * @author carl
 * @since 2023-01-26 11:18 PM
 **/
class LeeCode1663 {
    companion object {
        fun codeAsChar(code: Int) = ('a' + code - 1)

        fun getSmallestString(n: Int, k: Int): String {
            val res = StringBuilder()
            var total = k

            for (i in 1..n) {
                var diff = total - (n - i) * 26
                if (diff <= 1) {
                    diff = 1
                }

                total -= diff
                res.append(
                    codeAsChar(diff)
                )
            }
            return res.toString()
        }

    }
}

fun main(args: Array<String>) {
    println(codeAsChar(19))
    println(getSmallestString(5, 73))

}
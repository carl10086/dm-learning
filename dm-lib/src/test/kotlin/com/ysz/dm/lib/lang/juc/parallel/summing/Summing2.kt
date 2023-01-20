package com.ysz.dm.lib.lang.juc.parallel.summing

import java.util.*

/**
 * @author carl
 * @since 2023-01-18 3:52 PM
 **/
object Summing2 {

    fun basicSum(ia: LongArray): Long {
        var sum = 0L
        for (l in ia) {
            sum += l
        }
        return sum
    }

    const val SZ = 200_000_000L
    const val CHECK = SZ * (SZ + 1) / 2
}

fun main(args: Array<String>) {
    println(Summing2.CHECK)
    /*数组是预先就分配好了的,  这就意味着内存的限制.*/
    val la = LongArray((Summing2.SZ + 1L).toInt()) { it.toLong() }

    Summing.timeTest("Array Stream Sum", Summing2.CHECK) {
        Arrays.stream(la).sum()
    }

    Summing.timeTest("Parallel", Summing2.CHECK) {
        Arrays.stream(la).parallel().sum()
    }

    Summing.timeTest("Basic Sum", Summing2.CHECK) {
        Summing2.basicSum(la)
    }

    // 破坏性求和
    Summing.timeTest("parallelPrefix", Summing2.CHECK) {
        Arrays.parallelPrefix(la) { left, right ->
            left + right
        }
        /*数组的最后1个 . */
        la[la.size - 1]
    }

}
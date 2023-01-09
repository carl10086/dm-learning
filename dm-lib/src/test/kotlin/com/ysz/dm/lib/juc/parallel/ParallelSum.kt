package com.ysz.dm.lib.juc.parallel

import com.ysz.dm.lib.juc.timer.Timer
import java.util.function.LongSupplier

/**
 *     没有完成, 太无聊了， 不想写
 *@author carl.yu
 *@since 2023/1/10
 **/
class ParallelSum {
    companion object {
        const val SZ = 100_000_000

        /*高斯公式*/
        const val CHECK = SZ.toLong() * (SZ.toLong() + 1) / 2

        fun timeTest(id: String, checkValue: Long, operation: LongSupplier) {
            print("$id: ")

            val timer = Timer()
            val result = operation.asLong

            if (result == checkValue) println("${timer.duration()} ms")
            else println("result: $result , checkValue: $checkValue")
        }
    }
}



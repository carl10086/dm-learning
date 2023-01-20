package com.ysz.dm.lib.lang.juc.parallel.summing

import com.ysz.dm.lib.lang.juc.timer.Timer
import java.util.function.LongSupplier
import java.util.stream.LongStream

/**
 *     没有完成, 太无聊了， 不想写
 *@author carl.yu
 *@since 2023/1/10
 **/
class Summing {
    companion object {
        const val SZ = 100_000_000L

        /*高斯公式*/
        const val CHECK = SZ * (SZ + 1) / 2

        fun timeTest(id: String, checkValue: Long, operation: LongSupplier) {
            print("$id: ")

            val timer = Timer()
            val result = operation.asLong

            if (result == checkValue) println("${timer.duration()} ms")
            else println("result: $result , checkValue: $checkValue")
        }
    }
}

fun main() {
    println(Summing.CHECK)

    // 1. 这个方法最简单, 直接 stream . sum . 可以保证内存不挂的时候, 处理 1 亿级别的流
    Summing.timeTest("Sum Stream", Summing.CHECK) {
        LongStream.rangeClosed(0, Summing.SZ).sum()
    }

    // 2. 加了并行速度就飙升了
    Summing.timeTest("Sum Stream Parallel", Summing.CHECK) {
        LongStream.rangeClosed(0, Summing.SZ).parallel().sum()
    }

    // 3. 这个操作生成的流是 iterate, 每次循环都会调用 lambda 表达式
    Summing.timeTest("Sum Iterated", Summing.CHECK) {
        LongStream.iterate(0L) { it + 1L }.limit(Summing.SZ + 1).parallel().sum()
    }
}



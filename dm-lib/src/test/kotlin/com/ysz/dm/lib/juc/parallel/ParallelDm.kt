package com.ysz.dm.lib.juc.parallel

import com.ysz.dm.lib.juc.timer.Timer
import java.util.stream.LongStream
import kotlin.math.sqrt

/**
 * 使用分流器
 *@author carl.yu
 *@since 2023/1/10
 **/
class ParallelDm {

    companion object {
        const val COUNT = 100_000L

        /**
         * 判断一个数是不是 素数
         */
        fun isPrime(n: Long): Boolean =
            LongStream.rangeClosed(
                2, sqrt(n.toDouble()).toLong()
            ).noneMatch {
                n % it == 0L
            }
    }
}

fun main(args: Array<String>) {
    val timer = Timer()
    val primes = LongStream
        .iterate(2L) { it + 1 }
        /*注释下面的话速度可能会变慢 大概为 3倍时间*/
        .parallel()
        .filter { ParallelDm.isPrime(it) }
        .limit(ParallelDm.COUNT)
        .mapToObj { it.toString() }
        .toList()

    println(timer.duration())

    // 如果测试没有效果， 打开下面的注释， 这是为了欺骗编译器， 防止他优化掉， 什么都没有执行
//    Files.write(Paths.get("prims.txt"), primes, StandardOpenOption.CREATE)

}
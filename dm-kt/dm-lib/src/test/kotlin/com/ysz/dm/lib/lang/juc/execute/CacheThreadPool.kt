package com.ysz.dm.lib.lang.juc.execute

import java.util.concurrent.Executors

/**
 * @author carl
 * @since 2023-01-18 4:49 PM
 **/

fun main() {
    val exec = Executors.newCachedThreadPool()

    IntRange(0, 10).map(::NapTask).forEach { exec.execute(it) }

    exec.shutdown()

}
package com.ysz.dm.lib.juc.execute

import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/10
 **/
class MoreTasksAfterShutdownDm

fun main(args: Array<String>) {
    val exec = Executors.newSingleThreadExecutor()
    exec.execute(NapTask(1))
    exec.shutdown()

    /*此时会拒绝任务*/
    try {
        exec.execute(NapTask(99))
    } catch (e: RejectedExecutionException) {
        println(e)
    }
}
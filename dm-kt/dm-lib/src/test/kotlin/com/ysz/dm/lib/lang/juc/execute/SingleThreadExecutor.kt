package com.ysz.dm.lib.lang.juc.execute

import java.util.concurrent.Executors

/**
 *@author carl.yu
 *@since 2023/1/10
 **/
class SingleThreadExecutor

fun main(args: Array<String>) {
    val exec = Executors.newSingleThreadExecutor()

    (0..10).map { NapTask(it) }.forEach { exec.execute(it) }

    println("All tasks submitted")

    /*程序会在所有任务执行完之后结束*/
    /*同时 线程池不会再接收 新的任务*/
    /*继续调用会报错， 拒绝 RejectedException 异常*/
    exec.shutdown()

    /*这个不是必须的*/
    while (!exec.isTerminated) {
        println("${Thread.currentThread().name} awaiting termination")
        Nap(0.1)
    }
}
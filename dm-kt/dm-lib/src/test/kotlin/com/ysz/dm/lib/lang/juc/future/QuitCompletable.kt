package com.ysz.dm.lib.lang.juc.future

import com.ysz.dm.lib.lang.juc.execute.Nap
import com.ysz.dm.lib.lang.juc.execute.QuitAbleTask
import java.util.concurrent.CompletableFuture

/**
 * @author carl
 * @since 2023-01-20 12:38 AM
 **/

fun main(args: Array<String>) {
    val tasks = (1..150)
        .map(::QuitAbleTask)


    val futures =
        tasks.map {
            /*会自动开始, 可以传线程池*/
            CompletableFuture.runAsync(it)
        }

    Nap(1.0)

    tasks.forEach { it.quit() }


    /*不 join 到 main thread 会自动退出*/
    futures.forEach { it.join() }

}
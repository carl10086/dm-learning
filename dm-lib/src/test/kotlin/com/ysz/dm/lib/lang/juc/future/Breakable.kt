package com.ysz.dm.lib.lang.juc.future

import java.util.concurrent.CompletableFuture

/**
 * @author carl
 * @since 2023-01-20 2:16 AM
 **/
class Breakable(
    private val id: String,
    private var failCount: Int
) {
    override fun toString(): String {
        return "Breakable(id='$id', failCount=$failCount)"
    }

    companion object {
        fun work(b: Breakable): Breakable {
            if (--b.failCount == 0) {
                println("Throwing Exception for ${b.id}")
                throw RuntimeException("Breakable_${b.id} failed")
            } else {
                println(b)
            }

            return b
        }
    }
}

fun main(args: Array<String>) {
    val test = fun(id: String, failCount: Int): CompletableFuture<Breakable> {
        return CompletableFuture.completedFuture(Breakable(id, failCount))
            .thenApply(Breakable::work)
            .thenApply(Breakable::work)
            .thenApply(Breakable::work)
            .thenApply(Breakable::work)
    }

    // 1. 这个几个方都走到了 throwing , 但是没有抛异常
    test("A", 1)
    test("B", 2)
    test("C", 3)


    // 2. 得到了异常
    try {
        test("F", 2).get()
    } catch (e: Exception) {
        println("success catch exception:${e.message}")
    }


    // 3. 抛出异常的话
    println(test("G", 2).isCompletedExceptionally)
    println(test("F", 2).isDone)

    // 4.
    val cfi = CompletableFuture<Int>()
    println("cfi is done:${cfi.isCompletedExceptionally}")

    cfi.completeExceptionally(RuntimeException("forced"))

    try {
        cfi.get()
    } catch (e: Exception) {
        println("cfi catched exception:${e.message}")
        println("cfi is done:${cfi.isDone}, cfi is exception:${cfi.isCompletedExceptionally}")
    }
}